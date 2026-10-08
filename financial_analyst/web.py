"""Temporary access-code-protected browser testing over the existing agent."""
from __future__ import annotations

import argparse
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import math
import os
from pathlib import Path
import re
import threading

from .corpus_agent import CorpusToolAgent
from .corpus_cli import build_service, load_config, render_answer
from .inference import create_inference


MAX_BODY_BYTES=8192
MAX_QUESTION_CHARS=2000
ACCESS_CODE_ENV='FINANCIAL_ANALYST_ACCESS_CODE'

PAGE="""<!doctype html>
<html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Financial document analyst</title>
<style>body{font:16px system-ui;max-width:780px;margin:48px auto;padding:0 20px}label{display:block;margin:20px 0 8px}input,textarea,button{font:inherit;padding:10px;box-sizing:border-box}input,textarea{width:100%}button{margin-top:16px}#answer{white-space:pre-wrap;line-height:1.6;margin-top:28px}a{overflow-wrap:anywhere}</style>
<h1>Financial document analyst</h1><p>Ask a question about the reviewed reports.</p>
<form id="form"><label for="code">Private access code</label><input id="code" type="password" autocomplete="off" required>
<label for="question">Question</label><textarea id="question" rows="4" maxlength="2000" required></textarea>
<button id="submit">Ask</button></form><div id="answer" role="status" aria-live="polite"></div>
<script>
const form=document.getElementById('form'),result=document.getElementById('answer'),button=document.getElementById('submit');
function show(text){
  result.replaceChildren();
  const pattern=/https:\\/\\/[^\\s]+/g;let offset=0;
  for(const match of text.matchAll(pattern)){
    result.append(document.createTextNode(text.slice(offset,match.index)));
    try{const url=new URL(match[0]);if(url.protocol!=='https:')throw new Error();
      const link=document.createElement('a');link.href=url.href;link.textContent=match[0];link.target='_blank';link.rel='noopener noreferrer';result.append(link);
    }catch{result.append(document.createTextNode(match[0]));}
    offset=match.index+match[0].length;
  }
  result.append(document.createTextNode(text.slice(offset)));
}
form.addEventListener('submit',async event=>{
  event.preventDefault();button.disabled=true;show('Reading the reviewed sources…');
  try{
    const response=await fetch('/ask',{method:'POST',cache:'no-store',headers:{'Content-Type':'application/json','Authorization':'Bearer '+document.getElementById('code').value},body:JSON.stringify({question:document.getElementById('question').value})});
    const value=await response.json();show(value.answer||value.error||'No answer produced.');
  }catch{show('The connection could not complete. Try again after the current request finishes.');}
  finally{button.disabled=false;}
});
</script></html>"""


def _access_code(value):
    if not isinstance(value,str) or re.fullmatch(r'[A-Za-z0-9_-]{32,128}',value)is None:
        raise ValueError('A private high-entropy access code is required.')
    return value


def _positive_timeout(value):
    if isinstance(value,bool) or not isinstance(value,(int,float)) or not math.isfinite(value) or value<=0:
        raise ValueError('A finite positive timeout is required.')
    return value


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate request field.')
        result[key]=value
    return result


class AnalystHTTPServer(ThreadingHTTPServer):
    """One bounded agent operation at a time, including after HTTP timeout."""
    daemon_threads=True
    allow_reuse_address=True

    def __init__(self,agent,access_code,*,port=8765,answer_timeout=75,read_timeout=5,max_connections=8):
        self.access_code=_access_code(access_code)
        self.answer_timeout=_positive_timeout(answer_timeout)
        self.read_timeout=_positive_timeout(read_timeout)
        self.agent=agent
        self.operation_lock=threading.Lock()
        if type(max_connections)is not int or not 1<=max_connections<=32:
            raise ValueError('A bounded connection limit is required.')
        self.handler_slots=threading.BoundedSemaphore(max_connections)
        super().__init__(('127.0.0.1',port),AnalystHandler)

    def process_request(self,request,client_address):
        # Admit before ThreadingMixIn can spawn a handler, including incomplete
        # unauthenticated headers. Excess connections are closed immediately.
        if not self.handler_slots.acquire(blocking=False):
            self.shutdown_request(request)
            return
        try:
            super().process_request(request,client_address)
        except BaseException:
            self.handler_slots.release()
            raise

    def process_request_thread(self,request,client_address):
        try:
            super().process_request_thread(request,client_address)
        finally:
            self.handler_slots.release()

    def handle_error(self,request,client_address):
        # Never log request headers, paths, source data, or access credentials.
        pass

    def answer(self,question):
        if not self.operation_lock.acquire(blocking=False):
            return 429,{'error':'Another question is being processed. Try again after it finishes.'}
        completed=threading.Event()
        result={}
        def run():
            try:
                answer=self.agent.answer(question)
                result.update(status=answer.status,answer=render_answer(answer))
            except Exception:
                result.update(error='No answer produced. Check source integrity and provider availability.')
            finally:
                self.operation_lock.release()
                completed.set()
        try:
            threading.Thread(target=run,daemon=True).start()
        except Exception:
            self.operation_lock.release()
            return 503,{'error':'The question could not start.'}
        if not completed.wait(self.answer_timeout):
            return 504,{'error':'The request timed out. Wait for the current operation to finish before trying again.'}
        return (503 if 'error'in result else 200),result


class AnalystHandler(BaseHTTPRequestHandler):
    protocol_version='HTTP/1.1'

    def setup(self):
        super().setup()
        self.connection.settimeout(self.server.read_timeout)

    def log_message(self,*args):
        pass

    def _send(self,status,value,*,html=False):
        payload=value.encode('utf-8') if html else json.dumps(value,ensure_ascii=False).encode('utf-8')
        self.send_response(status)
        self.send_header('Content-Type','text/html; charset=utf-8' if html else 'application/json; charset=utf-8')
        self.send_header('Content-Length',str(len(payload)))
        self.send_header('Cache-Control','no-store')
        self.send_header('X-Content-Type-Options','nosniff')
        self.send_header('Referrer-Policy','no-referrer')
        self.send_header('Content-Security-Policy',"default-src 'none'; script-src 'unsafe-inline'; style-src 'unsafe-inline'; connect-src 'self'; frame-ancestors 'none'; base-uri 'none'; form-action 'none'")
        self.send_header('Connection','close')
        self.end_headers()
        self.close_connection=True
        try:self.wfile.write(payload)
        except OSError:pass

    def do_GET(self):
        if self.path=='/':self._send(200,PAGE,html=True)
        elif self.path=='/health':self._send(200,{'status':'ok'})
        else:self._send(404,{'error':'Not found.'})

    def do_POST(self):
        if self.path!='/ask':
            self._send(404,{'error':'Not found.'});return
        headers=self.headers.get_all('Authorization',[])
        expected=('Bearer '+self.server.access_code).encode('ascii')
        try:authorized=len(headers)==1 and hmac.compare_digest(headers[0].encode('ascii'),expected)
        except UnicodeError:authorized=False
        if not authorized:
            self._send(401,{'error':'A valid private access code is required.'});return
        lengths=self.headers.get_all('Content-Length',[])
        if self.headers.get('Transfer-Encoding')is not None or len(lengths)!=1 or not re.fullmatch(r'[0-9]{1,8}',lengths[0]):
            self._send(400,{'error':'A bounded JSON request is required.'});return
        length=int(lengths[0])
        if not 0<length<=MAX_BODY_BYTES:
            self._send(413,{'error':'Request body exceeds the allowed size.'});return
        if self.headers.get('Content-Type','').split(';',1)[0].strip().lower()!='application/json':
            self._send(415,{'error':'Use an application/json request.'});return
        try:
            raw=self.rfile.read(length)
            if len(raw)!=length:raise ValueError
            request=json.loads(raw.decode('utf-8'),object_pairs_hook=_unique)
            if not isinstance(request,dict) or set(request)!={'question'}:raise ValueError
            question=request['question']
            if not isinstance(question,str) or not question.strip() or len(question)>MAX_QUESTION_CHARS:raise ValueError
        except TimeoutError:
            self._send(408,{'error':'Request body timed out.'});return
        except (ValueError,UnicodeError,RecursionError,OSError):
            self._send(400,{'error':'Provide one nonblank question of at most 2,000 characters.'});return
        status,result=self.server.answer(question)
        self._send(status,result)

    def do_OPTIONS(self):
        self._send(405,{'error':'Method not allowed.'})


def main(argv=None):
    parser=argparse.ArgumentParser(description='Private browser testing of the reviewed financial analyst.')
    parser.add_argument('--config',type=Path,required=True,help='Explicit private corpus session configuration.')
    parser.add_argument('--port',type=int,default=8765)
    args=parser.parse_args(argv)
    inference=server=None
    try:
        code=_access_code(os.environ.get(ACCESS_CODE_ENV))
        if not 1<=args.port<=65535:raise ValueError
        config=load_config(args.config)
        if config['mode']!='live':raise ValueError('Browser testing requires real SQLite mode.')
        service=build_service(config,'live')
        inference=create_inference(config['provider'],model=config.get('model'),env_file=config.get('env_file'),
            base_url=config.get('base_url'),api_key_env=config.get('api_key_env'),timeout=30)
        server=AnalystHTTPServer(CorpusToolAgent(inference,service),code,port=args.port)
        print(f'Browser test server ready at http://127.0.0.1:{args.port}. Private access code required.',flush=True)
        server.serve_forever()
        return 0
    except KeyboardInterrupt:return 0
    except Exception:
        print('Browser test server could not start. Check the private access code, configuration, sources and provider access.')
        return 2
    finally:
        if server is not None:server.server_close()
        if inference is not None:
            try:inference.close()
            except Exception:pass


if __name__=='__main__':raise SystemExit(main())
