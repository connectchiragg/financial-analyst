"""CLI modes, session boundaries and source-preserving presentation."""
from contextlib import redirect_stdout
from dataclasses import replace
from decimal import Decimal
from io import StringIO
import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import Mock,patch

from financial_analyst import corpus_cli
from financial_analyst.corpus_adapter import FactFilter,import_reviewed
from tests.test_agent import FakeInference
from tests.test_corpus_adapter import create_corpus
from tests.test_corpus_agent import plan,decision


class CorpusCLITests(unittest.TestCase):
    def setUp(self):
        self.directory=TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        directory=Path(self.directory.name)
        pack,catalog,_,self.pdf=create_corpus(directory)
        self.db=directory/'reviewed.sqlite'
        self.config_path=directory/'session.json'
        self.config_path.write_text(json.dumps({'proof_pack':pack.name,'catalog':catalog.name,
            'database':self.db.name,'provider':'groq','env_file':'credentials-not-present.env'}))
        self.config=corpus_cli.load_config(self.config_path)

    def capture(self,*args):
        output=StringIO()
        with redirect_stdout(output):
            code=corpus_cli.main(['--config',str(self.config_path),*args])
        return code,output.getvalue()

    def test_import_and_coverage_execute_real_sqlite_without_inference(self):
        with patch.object(corpus_cli,'create_inference') as inference:
            code,text=self.capture('ingest')
            self.assertEqual(code,0)
            self.assertEqual(json.loads(text)['counts']['facts_added'],2)
            code,text=self.capture('coverage')
            self.assertEqual(code,0)
            result=json.loads(text)
            self.assertEqual(result['execution']['database'],'sqlite_corpus')
            self.assertEqual(result['companies'][0]['facts'],2)
            self.assertIn('not a completeness',result['coverage_claim'])
            inference.assert_not_called()

    def test_fixture_coverage_does_not_create_database(self):
        code,text=self.capture('--mode','fixture','coverage')
        self.assertEqual(code,0)
        self.assertFalse(self.db.exists())
        self.assertEqual(json.loads(text)['execution']['database'],'none')

    def test_missing_sqlite_and_tampered_source_fail_without_provider(self):
        with patch.object(corpus_cli,'create_inference') as inference:
            code,text=self.capture('ask','What was Example Pharma revenue?')
            self.assertEqual(code,2)
            self.assertFalse(self.db.exists())
            self.pdf.write_bytes(self.pdf.read_bytes()+b'tampered')
            code,text=self.capture('ingest')
            self.assertEqual(code,2)
            self.assertFalse(self.db.exists())
            inference.assert_not_called()

    def test_question_only_session_resets_state_closes_provider_and_keeps_exact_output(self):
        import_reviewed(self.config['proof_pack'],self.config['catalog'],self.db)
        inference=FakeInference(plan(),decision())
        inference.close=Mock()
        questions=iter(['','What was Example Pharma revenue in 1QFY27?','/quit'])
        output=[]
        with patch.object(corpus_cli,'create_inference',return_value=inference):
            code=corpus_cli.run_chat(self.config,input_fn=lambda prompt:next(questions),output_fn=output.append)
        self.assertEqual(code,0)
        self.assertEqual(output[0],'Reports available for 1 company. Ask about a company, period, or metric.')
        self.assertIn('Ask a question. Type /quit to exit.',output)
        self.assertIn('1,200.00000000000000000000',output[-1].replace('1200.','1,200.'))
        self.assertNotIn('sqlite_corpus',output[-1])
        self.assertNotIn('Original source labels',output[-1])
        self.assertIn('Sources:',output[-1])
        self.assertEqual(len(inference.calls),2)
        inference.close.assert_called_once()

    def test_config_relative_paths_resolve_from_config_and_unknown_fields_reject(self):
        self.assertEqual(self.config['database'],self.db.resolve())
        self.config_path.write_text('{"provider":"groq","provider":"groq"}')
        with self.assertRaises(ValueError):corpus_cli.load_config(self.config_path)
        self.config_path.write_text('{"private_token":"not-printable"}')
        code,text=self.capture('coverage')
        self.assertEqual(code,2)
        self.assertNotIn('not-printable',text)

    def test_plain_quote_render_retains_unknown_scope_and_source_conditions(self):
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.lookup(FactFilter(companies=('Example Pharma',)),('example-growth',))
        rendered=corpus_cli.render_answer(answer)
        self.assertNotIn('scope not specified',rendered)
        self.assertNotIn('standalone',rendered)
        self.assertNotIn('consolidated',rendered)
        self.assertIn(answer.claims[0].values['quote'],rendered)
        self.assertNotIn('Source provenance:',rendered)
        self.assertIn('Sources:',rendered)

    def test_plain_refusal_is_friendly_and_diagnostics_keep_the_reason(self):
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.refuse('Synthetic missing-context diagnostic.',used=False)
        self.assertEqual(corpus_cli.render_answer(answer),
            "I'm sorry, I don't have an answer to that from these documents.")
        self.assertIn('Synthetic missing-context diagnostic.',
            corpus_cli.render_answer(answer,diagnostics=True))

    def test_corpus_engine_and_mode_are_explicit_and_no_unknown_engine_falls_back(self):
        self.config_path.write_text(json.dumps({'engine':'corpus','mode':'fixture',
            'proof_pack':self.config['proof_pack'].name,'catalog':self.config['catalog'].name,
            'database':self.db.name,'provider':'groq'}))
        config=corpus_cli.load_config(self.config_path)
        self.assertEqual(config['engine'],'corpus')
        self.assertEqual(config['mode'],'fixture')
        with patch.object(corpus_cli,'create_inference') as inference:
            code,text=self.capture('coverage')
        self.assertEqual(code,0)
        self.assertEqual(json.loads(text)['execution']['database'],'none')
        inference.assert_not_called()
        self.assertFalse(self.db.exists())
        config['engine']='unknown'
        self.config_path.write_text(json.dumps({key:str(value) for key,value in config.items()}))
        with self.assertRaises(ValueError):corpus_cli.load_config(self.config_path)

    def test_corpus_chat_source_failure_does_not_create_provider_or_fallback(self):
        self.pdf.write_bytes(self.pdf.read_bytes()+b'tampered')
        output=[]
        with patch.object(corpus_cli,'create_inference') as factory:
            code=corpus_cli.run_chat(self.config,mode='fixture',input_fn=lambda _: '/quit',output_fn=output.append)
        self.assertEqual(code,2)
        factory.assert_not_called()
        self.assertIn('Chat could not start','\n'.join(output))
        self.assertFalse(self.db.exists())

    def test_corpus_session_cleanup_for_quit_eof_interrupt_and_input_failure(self):
        for ending in ('/quit',EOFError(),KeyboardInterrupt(),RuntimeError('SECRET_PROVIDER_DETAIL')):
            with self.subTest(ending=type(ending).__name__):
                inference=FakeInference()
                inference.close=Mock()
                output=[]
                def read(_):
                    if isinstance(ending,BaseException):raise ending
                    return ending
                with patch.object(corpus_cli,'create_inference',return_value=inference):
                    code=corpus_cli.run_chat(self.config,mode='fixture',input_fn=read,output_fn=output.append,diagnostics=True)
                self.assertEqual(code,2 if isinstance(ending,RuntimeError) else 0)
                inference.close.assert_called_once()
                self.assertFalse(inference.calls)
                self.assertNotIn('SECRET_PROVIDER_DETAIL','\n'.join(output))
                self.assertIn('database: none','\n'.join(output))
                self.assertIn('LLM: test_double','\n'.join(output))

    def test_failed_agent_construction_closes_provider_once_without_sensitive_output(self):
        inference=FakeInference();inference.close=Mock();output=[]
        with patch.object(corpus_cli,'create_inference',return_value=inference), \
             patch.object(corpus_cli,'CorpusToolAgent',side_effect=RuntimeError('SECRET_PROVIDER_DETAIL')):
            code=corpus_cli.run_chat(self.config,mode='fixture',output_fn=output.append)
        self.assertEqual(code,2)
        inference.close.assert_called_once()
        self.assertNotIn('SECRET_PROVIDER_DETAIL','\n'.join(output))

    def test_coverage_counts_actual_approved_rows_not_catalog_companies(self):
        pack_path=self.config['proof_pack']
        pack=json.loads(pack_path.read_text())
        for row in pack['facts']:
            row['review'].update(status='source_verified',provenance='source_verified_proof_pack',
                                 promotion=False,capabilities=[])
        pack_path.write_text(json.dumps(pack))
        service=corpus_cli.build_service(self.config,'fixture')
        self.assertEqual(service.adapter.companies(),('Example Pharma',))
        coverage=corpus_cli.coverage_details(service)
        self.assertEqual(coverage['companies'],[])
        output=[]
        with patch.object(corpus_cli,'create_inference') as factory:
            code=corpus_cli.run_chat(self.config,mode='fixture',output_fn=output.append)
        self.assertEqual(code,2)
        factory.assert_not_called()
        self.assertIn('No approved reviewed records','\n'.join(output))

    def test_valuation_reconciliation_retains_printed_target_and_discrepancies(self):
        from financial_analyst.service import Answer,Claim
        service=corpus_cli.build_service(self.config,'fixture')
        citation=service.adapter.resolve('valuation').citation
        computed_ev=Decimal('100')*Decimal('6.5')
        computed_per_share=Decimal('800')/Decimal('2')
        values={'company':'Example Pharma','period':'FY28','scope':None,'currency':'INR','unit':'million',
            'valuation_ebitda':Decimal('100'),'valuation_multiple':Decimal('6.5'),
            'computed_ev':computed_ev,'printed_ev':Decimal('651'),
            'ev_difference':computed_ev-Decimal('651'),'printed_cash':Decimal('149'),
            'printed_equity':Decimal('800'),'printed_shares_millions':Decimal('2'),
            'computed_per_share':computed_per_share,'printed_target':Decimal('390'),
            'per_share_difference':computed_per_share-Decimal('390'),
            'source_consistency':'unreconciled_printed_values',
            'calculation_basis':'same printed valuation table; accounting scope unspecified',
            'fact_ids':('synthetic-valuation',),'evidence_refs':('valuation',)}
        answer=Answer('answered',(Claim('source_table_reconciliation',values,('valuation',)),),
                      citations=(citation,),execution=service._execution())
        rendered=corpus_cli.render_answer(answer,diagnostics=True)
        self.assertIn('scope not specified; valuation reconciliation',rendered)
        self.assertIn('computed EV INR 650.0 million; printed EV INR 651 million',rendered)
        self.assertIn('difference (computed minus printed) INR -1.0 million',rendered)
        self.assertIn('Printed cash INR 149 million',rendered)
        self.assertIn('printed shares 2 million',rendered)
        self.assertIn('Computed equity per share INR 400.00; printed target INR 390 per share',rendered)
        self.assertIn('difference (computed minus printed target) INR 10.00 per share',rendered)
        self.assertIn('Printed source values do not reconcile exactly',rendered)
        self.assertIn('accounting scope unspecified',rendered)
        self.assertIn(citation.document_name,rendered)
        self.assertIn('Per-share calculations are rounded to two decimals',rendered)
        self.assertNotIn('Calculated percentages',rendered)
        self.assertNotIn('net sales',rendered)

    def test_printed_growth_is_labeled_as_source_reported(self):
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.lookup(FactFilter(companies=('Example Pharma',)),('example-growth',))
        claim=replace(answer.claims[0],values={**answer.claims[0].values,
            'quote':None,'metric':'revenue_yoy_growth','kind':'source_reported_growth',
            'value_text':'23.4','unit':'percent','currency':None})
        text=corpus_cli.render_answer(replace(answer,claims=(claim,)))
        self.assertIn('revenue yoy growth (source reported) 23.4 percent',text)
        self.assertNotIn('(actual)',text)

    def test_simple_margin_display_retains_full_quote_cross_sentence_conditions_and_answer(self):
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.lookup(FactFilter(companies=('Example Pharma',)),('example-growth',))
        quote=('Volume growth improved. Margin remains under pressure in 2Q and normalizes '
            'in 2H if input costs ease. \uf06e Further price hikes are under evaluation. '
            'Commodity costs ease only if geopolitical conditions stabilize. '
            'This depends on demand. That recovery is not assured. We estimate')
        claim=replace(answer.claims[0],values={**answer.claims[0].values,'metric':'margin_outlook','quote':quote})
        answer=replace(answer,claims=(claim,))
        original=json.dumps(corpus_cli._json_value(answer),sort_keys=True)
        rendered=corpus_cli.render_answer(answer)
        self.assertIn('Margin remains under pressure in 2Q and normalizes in 2H if input costs ease.',rendered)
        self.assertIn('Further price hikes are under evaluation.',rendered)
        self.assertIn('Commodity costs ease only if geopolitical conditions stabilize.',rendered)
        self.assertIn('Volume growth improved',rendered)
        self.assertIn('This depends on demand. That recovery is not assured.',rendered)
        self.assertIn('We estimate',rendered)
        self.assertNotIn('\uf06e',rendered)
        self.assertNotIn('Execution:',rendered)
        self.assertNotIn('source quotation:',rendered)
        body,sources=rendered.split('\n\nSources:\n')
        self.assertNotIn('https://',body)
        self.assertEqual(sources.count('synthetic-corpus.pdf'),1)
        self.assertEqual(sources.count('https://example.test/report.pdf'),1)
        self.assertEqual(json.dumps(corpus_cli._json_value(answer),sort_keys=True),original)
        self.assertEqual({ref for claim in answer.claims for ref in claim.evidence_refs},
            {citation.ref for citation in answer.citations})

    def test_simple_numeric_forecast_is_explicit_and_unknown_scope_is_not_inferred(self):
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.lookup(FactFilter(companies=('Example Pharma',)),('example-revenue',))
        claim=replace(answer.claims[0],values={**answer.claims[0].values,'kind':'broker_forecast','scope':None})
        answer=replace(answer,claims=(claim,))
        rendered=corpus_cli.render_answer(answer)
        self.assertIn('net sales (broker forecast)',rendered)
        self.assertIn('INR 1200.00000000000000000000 million',rendered)
        self.assertNotIn('consolidated',rendered)
        self.assertNotIn('actual',rendered)
        self.assertNotIn('Original source labels',rendered)
        self.assertIn('Original source labels',corpus_cli.render_answer(answer,diagnostics=True))

    def test_cli_diagnostics_is_explicit_and_keeps_source_labels(self):
        inference=FakeInference(plan(),decision())
        inference.close=Mock()
        with patch.object(corpus_cli,'create_inference',return_value=inference):
            code,text=self.capture('--mode','fixture','--diagnostics','ask',
                'What was Example Pharma revenue in 1QFY27?')
        self.assertEqual(code,0)
        self.assertIn('Execution:',text)
        self.assertIn('Original source labels',text)

    def test_table_snapshot_is_preserved_with_other_requested_same_company_metric(self):
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.lookup(FactFilter(companies=('Example Pharma',)),
            ('example-revenue','example-growth'))
        row_text='Net Sales 900 1,000 1,100 1,200'
        row=replace(answer.claims[0],values={**answer.claims[0].values,
            'quote':row_text,'value_text':None})
        answer=replace(answer,claims=(row,answer.claims[1]))
        original=json.dumps(corpus_cli._json_value(answer),sort_keys=True)
        rendered=corpus_cli.render_answer(answer)
        self.assertIn(row_text,rendered)
        self.assertIn(answer.claims[1].values['quote'],rendered)
        self.assertIn(row_text,corpus_cli.render_answer(answer,diagnostics=True))
        self.assertEqual(json.dumps(corpus_cli._json_value(answer),sort_keys=True),original)
        self.assertIn('Sources:',rendered)

        # Its own only answer, or a different company's only answer, survives.
        row_answer=service.lookup(FactFilter(companies=('Example Pharma',)),('example-revenue',))
        self.assertIn(row_text,corpus_cli.render_answer(replace(row_answer,claims=(row,))))
        other_row=replace(row,values={**row.values,'company':'Other Example'})
        self.assertIn('Other Example: '+row_text,
            corpus_cli.render_answer(replace(answer,claims=(other_row,answer.claims[1]))))

    def test_numerical_prose_and_margin_conditions_remain_complete(self):
        prose=('Margins were 21% in 1Q and 23% in 2Q. Further price hikes remain '
            'under evaluation, and costs ease only if geopolitical conditions stabilize.')
        service=corpus_cli.build_service(self.config,'fixture')
        answer=service.lookup(FactFilter(companies=('Example Pharma',)),('example-growth',))
        claim=replace(answer.claims[0],values={**answer.claims[0].values,
            'metric':'margin_outlook','quote':prose})
        rendered=corpus_cli.render_answer(replace(answer,claims=(claim,)))
        self.assertIn('Margins were 21% in 1Q and 23% in 2Q.',rendered)
        self.assertIn('costs ease only if geopolitical conditions stabilize.',rendered)
        self.assertIn(prose,rendered)


if __name__=='__main__':unittest.main()
