"""Real loopback HTTP with synthetic answers, never providers/private files."""
from contextlib import redirect_stdout
import http.client
from io import StringIO
import json
import os
import socket
import threading
import time
import unittest
from unittest.mock import Mock,patch

from financial_analyst import web
from financial_analyst.adapters import Citation
from financial_analyst.service import Answer,Claim


CODE='synthetic-test-access-code-0123456789'


class SyntheticAgent:
    def __init__(self):
        self.calls=[]
        self.error=None
        self.started=threading.Event()
        self.release=None

    def answer(self,question):
        self.calls.append(question)
        self.started.set()
        if self.release is not None:self.release.wait(2)
        if self.error is not None:raise self.error
        if question=='unsupported':return Answer('refused',reason='Synthetic evidence does not support this question.')
        claim=Claim('source_fact',{'company':'Example Pharma','period':'1QFY27','scope':None,
            'metric':'operating_commentary','quote':'Synthetic source condition: costs ease only if demand stabilizes.'},('synthetic-ref',))
        return Answer('answered',(claim,),citations=(Citation('synthetic-ref','synthetic.pdf',1,
            'Synthetic statement','https://example.test/synthetic.pdf'),))


class BrowserHTTPTests(unittest.TestCase):
    def setUp(self):
        self.agent=SyntheticAgent()
        self.server=web.AnalystHTTPServer(self.agent,CODE,port=0,answer_timeout=1,read_timeout=.1)
        self.thread=threading.Thread(target=self.server.serve_forever,kwargs={'poll_interval':.01},daemon=True)
        self.thread.start()
        self.addCleanup(self.close)

    def close(self):
        if self.agent.release is not None:self.agent.release.set()
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(2)

    def request(self,path='/ask',*,method='POST',body=None,headers=None):
        connection=http.client.HTTPConnection('127.0.0.1',self.server.server_port,timeout=2)
        selected={'Content-Type':'application/json','Authorization':'Bearer '+CODE}
        if headers is not None:selected=headers
        if body is None:body=json.dumps({'question':'What does the source say?'})
        try:
            connection.request(method,path,body=body if method=='POST'else None,headers=selected)
            response=connection.getresponse()
            return response.status,dict(response.getheaders()),response.read().decode()
        finally:connection.close()

    def test_unauthorized_requests_never_call_agent_or_leak_code(self):
        for headers in ({},{'Authorization':'Bearer wrong'},{'Authorization':'Basic '+CODE}):
            status,_,body=self.request(headers=headers)
            self.assertEqual(status,401)
            self.assertNotIn(CODE,body)
        self.assertEqual(self.agent.calls,[])

    def test_authorized_response_preserves_refusal_and_cited_plain_answer(self):
        status,headers,body=self.request()
        self.assertEqual(status,200)
        answer=json.loads(body)
        self.assertEqual(answer['status'],'answered')
        self.assertIn('costs ease only if demand stabilizes.',answer['answer'])
        self.assertIn('Sources:',answer['answer'])
        self.assertIn('https://example.test/synthetic.pdf',answer['answer'])
        self.assertNotIn('Execution:',answer['answer'])
        self.assertEqual(headers['Cache-Control'],'no-store')
        status,_,body=self.request(body=json.dumps({'question':'unsupported'}))
        self.assertEqual(status,200)
        self.assertEqual(json.loads(body)['status'],'refused')
        self.assertIn('Unable to answer:',json.loads(body)['answer'])

    def test_page_health_no_files_no_cors_and_safe_browser_rendering(self):
        for path in ('/','/health','/private.env','/.local/analyst.sqlite','/../private.env'):
            status,headers,body=self.request(path,method='GET',headers={})
            self.assertEqual(status,200 if path in ('/','/health')else 404)
            self.assertNotIn(CODE,body)
            self.assertNotIn('Access-Control-Allow-Origin',headers)
        status,_,body=self.request('/health',method='GET',headers={})
        self.assertEqual(json.loads(body),{'status':'ok'})
        self.assertNotIn('innerHTML',web.PAGE)
        self.assertNotIn('localStorage',web.PAGE)
        self.assertIn('createTextNode',web.PAGE)
        self.assertIn("if(url.protocol!=='https:')",web.PAGE)
        self.assertEqual(self.agent.calls,[])

    def test_nested_json_and_unsupported_transfer_encoding_are_rejected(self):
        body='{"question":'+'['*1500+'"x"'+']'*1500+'}'
        status,_,_=self.request(body=body)
        self.assertEqual(status,400)
        status,_,_=self.request(headers={'Authorization':'Bearer '+CODE,
            'Content-Type':'application/json','Transfer-Encoding':'chunked'})
        self.assertEqual(status,400)
        self.assertEqual(self.agent.calls,[])

    def test_malformed_duplicate_extra_and_unbounded_inputs_do_not_call_agent(self):
        for body in ('not JSON','[]','{"question":"x","question":"y"}',
                '{"question":"x","url":"https://example.test"}',
                json.dumps({'question':False}),json.dumps({'question':' '}),
                json.dumps({'question':'x'*2001})):
            with self.subTest(body_length=len(body)):
                status,_,_=self.request(body=body)
                self.assertEqual(status,400)
        status,_,_=self.request(body='x'*(web.MAX_BODY_BYTES+1))
        self.assertEqual(status,413)
        status,_,_=self.request(headers={'Authorization':'Bearer '+CODE,'Content-Type':'text/plain'})
        self.assertEqual(status,415)
        self.assertEqual(self.agent.calls,[])

    def test_busy_requests_are_rejected_without_parallel_agent_calls(self):
        self.agent.release=threading.Event()
        result=[]
        first=threading.Thread(target=lambda:result.append(self.request()))
        first.start()
        self.assertTrue(self.agent.started.wait(1))
        status,_,body=self.request()
        self.assertEqual(status,429)
        self.assertIn('Another question',body)
        self.assertEqual(len(self.agent.calls),1)
        self.agent.release.set()
        first.join(2)
        self.assertEqual(result[0][0],200)

    def test_timeout_keeps_operation_busy_until_it_really_finishes(self):
        self.server.answer_timeout=.03
        self.agent.release=threading.Event()
        status,_,_=self.request()
        self.assertEqual(status,504)
        status,_,_=self.request()
        self.assertEqual(status,429)
        self.assertEqual(len(self.agent.calls),1)
        self.agent.release.set()
        deadline=time.monotonic()+1
        while self.server.operation_lock.locked()and time.monotonic()<deadline:time.sleep(.005)
        self.assertFalse(self.server.operation_lock.locked())
        status,_,_=self.request()
        self.assertEqual(status,200)

    def test_body_read_timeout_never_calls_agent(self):
        with socket.create_connection(('127.0.0.1',self.server.server_port),timeout=2)as connection:
            wire=('POST /ask HTTP/1.1\r\nHost: localhost\r\nAuthorization: Bearer '+CODE+
                '\r\nContent-Type: application/json\r\nContent-Length: 100\r\n\r\n')
            connection.sendall(wire.encode())
            self.assertIn(b'408',connection.recv(4096).split(b'\r\n',1)[0])
        self.assertEqual(self.agent.calls,[])

    def test_incomplete_unauthenticated_headers_have_bounded_admission(self):
        self.server.read_timeout=.25
        sockets=[]
        with patch.object(self.server,'process_request_thread',wraps=self.server.process_request_thread)as handlers:
            try:
                for _ in range(8):
                    connection=socket.create_connection(('127.0.0.1',self.server.server_port),timeout=1)
                    connection.sendall(b'GET / HTTP/1.1\r\n')
                    sockets.append(connection)
                deadline=time.monotonic()+.2
                while handlers.call_count<8 and time.monotonic()<deadline:time.sleep(.002)
                self.assertEqual(handlers.call_count,8)
                with socket.create_connection(('127.0.0.1',self.server.server_port),timeout=1)as extra:
                    extra.sendall(b'GET / HTTP/1.1\r\n')
                    try:closed=extra.recv(1)==b''
                    except ConnectionResetError:closed=True
                    self.assertTrue(closed)
                self.assertEqual(handlers.call_count,8)
                time.sleep(.3)
                self.assertEqual(self.request('/health',method='GET',headers={})[0],200)
                self.assertEqual(handlers.call_count,9)
            finally:
                for connection in sockets:connection.close()
        self.assertEqual(self.agent.calls,[])

    def test_failed_handler_spawn_releases_admission_slot(self):
        with patch('socketserver.ThreadingMixIn.process_request',side_effect=RuntimeError('synthetic spawn failure')):
            with socket.socket()as request:
                with self.assertRaises(RuntimeError):
                    self.server.process_request(request,('127.0.0.1',1))
        available=0
        while self.server.handler_slots.acquire(blocking=False):available+=1
        self.assertEqual(available,8)
        for _ in range(available):self.server.handler_slots.release()

    def test_errors_are_generic_and_release_the_operation_lock(self):
        self.agent.error=RuntimeError('PRIVATE_SOURCE_AND_CREDENTIAL_DETAIL')
        status,_,body=self.request()
        self.assertEqual(status,503)
        self.assertNotIn('PRIVATE_SOURCE',body)
        self.assertFalse(self.server.operation_lock.locked())

    def test_access_code_is_required_before_loading_private_configuration(self):
        for value in (None,'short','bad code'*8):
            with self.subTest(value_type=type(value).__name__),patch.dict(os.environ,{},clear=True), \
                    patch.object(web,'load_config')as config,patch.object(web,'create_inference')as provider, \
                    redirect_stdout(StringIO()):
                if value is not None:os.environ[web.ACCESS_CODE_ENV]=value
                self.assertEqual(web.main(['--config','never-read.json']),2)
                config.assert_not_called()
                provider.assert_not_called()

    def test_main_uses_real_mode_and_closes_the_provider(self):
        server=Mock()
        server.serve_forever.side_effect=KeyboardInterrupt
        provider=Mock()
        config={'mode':'live','provider':'openai-compatible','model':'synthetic-model'}
        with patch.dict(os.environ,{web.ACCESS_CODE_ENV:CODE}),patch.object(web,'load_config',return_value=config), \
                patch.object(web,'build_service')as service,patch.object(web,'create_inference',return_value=provider), \
                patch.object(web,'CorpusToolAgent')as agent,patch.object(web,'AnalystHTTPServer',return_value=server)as factory, \
                redirect_stdout(StringIO())as output:
            self.assertEqual(web.main(['--config','explicit-private.json']),0)
        service.assert_called_once_with(config,'live')
        agent.assert_called_once_with(provider,service.return_value)
        factory.assert_called_once_with(agent.return_value,CODE,port=8765)
        server.server_close.assert_called_once()
        provider.close.assert_called_once()
        self.assertNotIn(CODE,output.getvalue())


if __name__=='__main__':unittest.main()
