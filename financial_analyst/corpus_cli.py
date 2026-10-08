"""Reviewed cross-company CLI with explicit provider and SQLite boundaries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path

from .cli import _citation_text, _json_value, _number
from .corpus_adapter import FactFilter, ReviewedCorpusAdapter, SQLiteCorpusAdapter, import_reviewed
from .corpus_agent import CorpusToolAgent
from .corpus_service import CorpusService
from .inference import create_inference


DEFAULT_CONFIG=Path('.local/corpus-config.json')
CONFIG_FIELDS={'proof_pack','catalog','database','provider','model','env_file','base_url','api_key_env','engine','mode'}


def _unique(pairs):
    result={}
    for key,value in pairs:
        if key in result:raise ValueError('Duplicate configuration field.')
        result[key]=value
    return result


def load_config(path):
    path=Path(path).expanduser().resolve()
    config=json.loads(path.read_text(encoding='utf-8'),object_pairs_hook=_unique)
    if not isinstance(config,dict) or set(config)-CONFIG_FIELDS:
        raise ValueError('Unsupported corpus configuration fields.')
    if config.get('engine','corpus')!='corpus':
        raise ValueError('Unsupported corpus engine.')
    config['mode']=config.get('mode','live')
    if config['mode'] not in {'fixture','live'}:
        raise ValueError('Unsupported corpus execution mode.')
    for field in ('proof_pack','catalog','database','provider'):
        if not isinstance(config.get(field),str) or not config[field].strip():
            raise ValueError('Corpus configuration requires source and provider bindings.')
    if config['provider'] not in {'groq','mistral','openai-compatible'}:
        raise ValueError('Unknown inference provider.')
    for field in ('model','base_url','api_key_env'):
        if field in config and (not isinstance(config[field],str) or not config[field].strip()):
            raise ValueError('Invalid inference configuration.')
    for field in ('proof_pack','catalog','database','env_file'):
        if field not in config:continue
        if not isinstance(config[field],str) or not config[field].strip():
            raise ValueError('Invalid private file binding.')
        selected=Path(config[field]).expanduser()
        config[field]=selected if selected.is_absolute() else path.parent/selected
    return config


def build_service(config,mode='live'):
    validator=ReviewedCorpusAdapter(config['proof_pack'],config['catalog'])
    if mode=='fixture':return CorpusService(validator)
    if mode!='live':raise ValueError('Unknown corpus execution mode.')
    return CorpusService(SQLiteCorpusAdapter(config['database'],validator))


def coverage_details(service):
    """Count only approved records actually returned by the selected adapter."""
    facts=service.candidates(FactFilter())
    companies=sorted({fact.company for fact in facts})
    return {'status':'coverage','companies':[
        {'company':company,'metrics':sorted({fact.metric for fact in facts if fact.company==company}),
         'facts':sum(fact.company==company for fact in facts),
         'analytical_facts':sum(fact.company==company and 'analytical_context' in fact.capabilities for fact in facts)}
        for company in companies],
        'execution':service._execution(),
        'coverage_claim':'Available reviewed records; not a completeness or accuracy guarantee.'}


def _coverage_banner(coverage,inference):
    companies=coverage['companies']
    metrics=sorted({metric for company in companies for metric in company['metrics']})
    execution=coverage['execution']
    company_label='company' if len(companies)==1 else 'companies'
    metric_label='metric' if len(metrics)==1 else 'metrics'
    return (
        f"Available reviewed records: {len(companies)} {company_label}; {len(metrics)} {metric_label}.",
        'Companies: '+', '.join(company['company'] for company in companies)+'.',
        'Metrics: '+', '.join(metric.replace('_',' ') for metric in metrics)+'.',
        'Coverage is limited to reviewed records; periods, scopes and analytical support vary.',
        f"Execution: {execution['mode']}; database: {execution['database']}; "
        f"LLM: {inference.mode} {inference.provider} ({inference.model}).",
    )


def render_answer(answer):
    execution=answer.execution
    llm=execution.get('llm',{})
    inference='none' if llm.get('mode')=='not_called' else f"{llm.get('mode')} {llm.get('provider')} ({llm.get('model')})"
    lines=[f"Execution: {execution.get('mode')}; analytics: {execution.get('analytics')}; "
           f"LLM: {inference}; database: {execution.get('database')}; "
           f"retrieval: {execution.get('retrieval')}." ]
    if execution.get('seed_provenance'):
        lines.append('Source provenance: '+', '.join(execution['seed_provenance'])+'.')
    if answer.status!='answered':
        return '\n'.join((*lines,'Refused: '+str(answer.reason)))
    for claim in answer.claims:
        values=claim.values
        cite=_citation_text(claim.evidence_refs,answer.citations)
        company=values['company']
        period=values.get('period') or 'period not specified'
        scope=values.get('scope') or 'scope not specified'
        default_metric='valuation reconciliation' if claim.kind=='source_table_reconciliation' else 'net_sales'
        metric=values.get('metric',default_metric).replace('_',' ')
        prefix=f'{company}; {period}; {scope}; {metric}: '
        if claim.kind=='source_fact':
            if values['quote'] is not None:
                lines.append(prefix+'source quotation: '+values['quote']+cite)
            else:
                context=' '.join(str(values[key]) for key in ('currency','unit','kind') if values.get(key) is not None)
                lines.append(prefix+str(values['value_text'])+' '+context+cite)
                lines.append('Original source labels: '+json.dumps(values['raw_labels'],ensure_ascii=False,sort_keys=True)+'.')
        elif claim.kind=='metric_comparison':
            delta=values['delta_millions']
            label='above' if delta>0 else 'below' if delta<0 else 'equal to'
            lines.append(prefix+f"actual {values['currency']} {_number(values['actual_millions'])} million; "
                f"broker estimate {_number(values['estimate_millions'])} million; {label} estimate by "
                f"{_number(delta.copy_abs())} million ({_number(values['variance_percent'].copy_abs(),2)}%)."+cite)
        elif claim.kind in {'metric_estimate_change','metric_yoy_change'}:
            basis='broker estimate' if claim.kind=='metric_estimate_change' else values['reference_period']+' actual'
            currency=(values.get('currency')+' ') if values.get('currency') else ''
            lines.append(prefix+f"actual {currency}{_number(values['actual'])} {values['unit']}; "
                f"{basis} {_number(values['reference'])} {values['unit']}; "
                f"change {_number(values['delta'])} {values['delta_unit']}; "
                f"relative change {_number(values['relative_percent'],2)}%."+cite)
            if values['basis_points'] is not None:
                lines.append('Margin/rate change: '+_number(values['basis_points'])+' basis points.')
            lines.append(f"Source units retained: actual {values['actual_source_unit']}; reference {values['reference_source_unit']}.")
        elif claim.kind=='ranked_revenue_yoy':
            lines.append(f"{values['rank']}. "+prefix+f"{_number(values['yoy_percent'],2)}% YoY versus "
                f"{values['prior_period']}; actual {values['currency']} {_number(values['actual_millions'])} million; "
                f"prior {_number(values['prior_millions'])} million."+cite)
        elif claim.kind=='source_table_reconciliation':
            currency=values['currency']
            unit=values['unit']
            lines.append(prefix+f"printed EBITDA {currency} {_number(values['valuation_ebitda'])} {unit} "
                f"× {_number(values['valuation_multiple'])}x = computed EV {currency} {_number(values['computed_ev'])} {unit}; "
                f"printed EV {currency} {_number(values['printed_ev'])} {unit}; "
                f"difference (computed minus printed) {currency} {_number(values['ev_difference'])} {unit}."+cite)
            lines.append(f"Printed cash {currency} {_number(values['printed_cash'])} {unit}; "
                f"printed equity {currency} {_number(values['printed_equity'])} {unit}; "
                f"printed shares {_number(values['printed_shares_millions'])} million. "
                f"Computed equity per share {currency} {_number(values['computed_per_share'],2)}; "
                f"printed target {currency} {_number(values['printed_target'])} per share; "
                f"difference (computed minus printed target) {currency} {_number(values['per_share_difference'],2)} per share."+cite)
            consistency=values['source_consistency']
            if consistency=='unreconciled_printed_values':
                lines.append('Printed source values do not reconcile exactly.')
            elif consistency=='arithmetic_matches':
                lines.append('Printed source values match these calculations.')
            else:
                raise ValueError('Unknown source reconciliation status cannot be rendered.')
            lines.append('Calculation basis: '+values['calculation_basis']+'.')
        else:
            raise ValueError('Unknown canonical claim cannot be rendered.')
    percentage_kinds={'metric_comparison','metric_estimate_change','metric_yoy_change','ranked_revenue_yoy'}
    if any(claim.kind in percentage_kinds for claim in answer.claims):
        lines.append('Calculated percentages are rounded to two decimals for display.')
    if any(claim.kind=='source_table_reconciliation' for claim in answer.claims):
        lines.append('Per-share calculations are rounded to two decimals for display; printed values are retained.')
    return '\n\n'.join(lines)


def run_chat(config,mode='live',input_fn=input,output_fn=print):
    inference=None
    try:
        service=build_service(config,mode)
        coverage=coverage_details(service)
        if not coverage['companies']:
            output_fn('No approved reviewed records are available for this session.')
            return 2
        inference=create_inference(config['provider'],model=config.get('model'),env_file=config.get('env_file'),
                                   base_url=config.get('base_url'),api_key_env=config.get('api_key_env'),timeout=30)
        agent=CorpusToolAgent(inference,service)
        for line in _coverage_banner(coverage,inference):
            output_fn(line)
        output_fn('Ask a question. Type /quit to exit.')
        while True:
            try:question=input_fn('Question: ')
            except (EOFError,KeyboardInterrupt):return 0
            except Exception:
                output_fn('The question could not be read. The session has ended.')
                return 2
            if not isinstance(question,str) or not question.strip():continue
            if question.strip().casefold()=='/quit':return 0
            try:output_fn(render_answer(agent.answer(question)))
            except KeyboardInterrupt:return 0
            except Exception:output_fn('No answer produced. Check source integrity and provider availability.')
    except KeyboardInterrupt:
        return 0
    except Exception:
        output_fn('Chat could not start. Check the private configuration, reviewed sources, SQLite integrity and provider access.')
        return 2
    finally:
        if inference is not None:
            try:inference.close()
            except Exception:pass


def main(argv=None):
    parser=argparse.ArgumentParser(description='Source-reviewed cross-company financial analyst.')
    parser.add_argument('--config',type=Path,default=DEFAULT_CONFIG)
    parser.add_argument('--mode',choices=('fixture','live'))
    parser.add_argument('--format',choices=('text','json'),default='text')
    commands=parser.add_subparsers(dest='command')
    commands.add_parser('chat')
    commands.add_parser('ingest',help='Import explicitly approved source-proof records transactionally.')
    commands.add_parser('coverage',help='Show available reviewed metrics and context; no inference.')
    ask=commands.add_parser('ask');ask.add_argument('question')
    try:
        args=parser.parse_args(argv)
        config=load_config(args.config)
        mode=args.mode or config['mode']
        if args.command=='ingest':
            if mode!='live':raise ValueError('Reviewed ingestion requires live SQLite mode.')
            result={'status':'ingested','counts':asdict(import_reviewed(config['proof_pack'],config['catalog'],config['database'])),
                    'execution':{'database':'sqlite_corpus','no_llm':True,'provenance':'reviewed_proof_pack'}}
        elif args.command=='coverage':
            service=build_service(config,mode)
            result=coverage_details(service)
        elif args.command=='ask':
            service=build_service(config,mode)
            inference=create_inference(config['provider'],model=config.get('model'),env_file=config.get('env_file'),
                                       base_url=config.get('base_url'),api_key_env=config.get('api_key_env'),timeout=30)
            try:result=CorpusToolAgent(inference,service).answer(args.question)
            finally:inference.close()
            print(json.dumps(_json_value(result),indent=2,ensure_ascii=False) if args.format=='json' else render_answer(result))
            return 0 if result.status=='answered' else 1
        else:
            return run_chat(config,mode)
        print(json.dumps(_json_value(result),indent=2,ensure_ascii=False))
        return 0
    except KeyboardInterrupt:return 0
    except Exception:
        print('Corpus operation failed. Check private configuration, reviewed source files, SQLite integrity and provider availability.')
        return 2


if __name__=='__main__':raise SystemExit(main())
