"""Reviewed cross-company CLI with explicit provider and SQLite boundaries."""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import re

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


def render_diagnostic_answer(answer):
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


def _display_quote(quote):
    # Conditions can span sentences. Only normalize layout, never select or
    # discard canonical source content in the presentation layer.
    return ' '.join(re.sub(r'[\uf06e\u25a0\u2022]',' ',quote).split())


def _source_list(answer):
    refs=tuple(dict.fromkeys(ref for claim in answer.claims for ref in claim.evidence_refs))
    by_ref={citation.ref:citation for citation in answer.citations}
    if set(refs)!=set(by_ref):
        raise ValueError('Rendered claims require closed source citations.')
    grouped={}
    for ref in refs:
        citation=by_ref[ref]
        key=citation.link or citation.document_name
        group=grouped.setdefault(key,{'name':citation.document_name,'link':citation.link,'pages':set()})
        group['pages'].add(citation.page)
    lines=[]
    for group in grouped.values():
        pages=', '.join(str(page)for page in sorted(group['pages']))
        line=f"- {group['name']}, pp. {pages}"
        if group['link']:line+=': '+group['link']
        lines.append(line)
    return 'Sources:\n'+'\n'.join(lines)


def render_answer(answer,*,diagnostics=False):
    """Plain canonical answers; detailed execution remains an opt-in view."""
    if diagnostics:return render_diagnostic_answer(answer)
    if answer.status!='answered':return "I'm sorry, I don't have an answer to that from these documents."
    lines=[]
    multiple_companies=len({claim.values['company']for claim in answer.claims})>1
    # Share only identical, explicit context. Mixed scopes/periods and ranked
    # rows keep their own labels; source quotations retain their full wording.
    contexts={tuple(claim.values.get(key) for key in ('company','period','scope'))
        for claim in answer.claims}
    shared_context=(len(answer.claims)>1 and len(contexts)==1 and all(
        claim.kind!='ranked_revenue_yoy' and not (
            claim.kind=='source_fact' and claim.values['quote'] is not None)
        for claim in answer.claims))
    for claim in answer.claims:
        value=claim.values
        context=' '.join(str(value[key])for key in ('company','period','scope') if value.get(key))
        prefix='' if shared_context else context+': '
        metric=value.get('metric','net_sales').replace('_',' ')
        if claim.kind=='source_fact':
            if value['quote'] is not None:
                quote=_display_quote(value['quote'])
                lines.append((value['company']+': ' if multiple_companies else '')+quote)
            else:
                status={'reported_actual':'actual','broker_estimate':'broker estimate',
                        'broker_forecast':'broker forecast',
                        'source_reported_growth':'source reported',
                        'broker_valuation':'broker valuation'}.get(value.get('kind'))
                amount=' '.join(str(value[key])for key in ('currency','value_text','unit') if value.get(key)is not None)
                label=metric+(' ('+status+')' if status else '')
                lines.append(f'{prefix}{label} {amount}.')
        elif claim.kind=='metric_comparison':
            delta=value['delta_millions']
            direction='above' if delta>0 else 'below' if delta<0 else 'equal to'
            lines.append(f"{prefix}{metric} actual {value['currency']} {_number(value['actual_millions'])} million "
                f"versus broker estimate {_number(value['estimate_millions'])} million; {direction} estimate by "
                f"{_number(delta.copy_abs())} million ({_number(value['variance_percent'].copy_abs(),2)}%).")
        elif claim.kind in {'metric_estimate_change','metric_yoy_change'}:
            reference='broker estimate' if claim.kind=='metric_estimate_change' else value['reference_period']+' actual'
            currency=(value.get('currency')+' ') if value.get('currency') else ''
            lines.append(f"{prefix}{metric} actual {currency}{_number(value['actual'])} {value['unit']} "
                f"versus {reference} {_number(value['reference'])} {value['unit']}; "
                f"change {_number(value['delta'])} {value['delta_unit']} ({_number(value['relative_percent'],2)}%).")
            if value['basis_points']is not None:
                lines.append(f"Margin/rate change: {_number(value['basis_points'])} basis points.")
        elif claim.kind=='ranked_revenue_yoy':
            lines.append(f"{value['rank']}. {context}: revenue grew {_number(value['yoy_percent'],2)}% YoY "
                f"versus {value['prior_period']}; actual {value['currency']} {_number(value['actual_millions'])} million "
                f"versus prior {_number(value['prior_millions'])} million.")
        elif claim.kind=='source_table_reconciliation':
            currency,unit=value['currency'],value['unit']
            lines.append(f"{prefix}EBITDA {currency} {_number(value['valuation_ebitda'])} {unit} "
                f"× {_number(value['valuation_multiple'])}x gives EV {_number(value['computed_ev'])} {unit}; "
                f"the printed EV is {_number(value['printed_ev'])} {unit} "
                f"(computed minus printed: {_number(value['ev_difference'])} {unit}).")
            lines.append(f"Printed cash: {currency} {_number(value['printed_cash'])} {unit}; "
                f"equity: {_number(value['printed_equity'])} {unit}; shares: {_number(value['printed_shares_millions'])} million. "
                f"Equity divided by shares gives {currency} {_number(value['computed_per_share'],2)} per share, "
                f"versus the printed target of {_number(value['printed_target'])} "
                f"(difference: {_number(value['per_share_difference'],2)} per share).")
            if value['source_consistency']=='unreconciled_printed_values':
                lines.append('The printed values do not reconcile exactly; accounting scope is unspecified.')
            elif value['source_consistency']=='arithmetic_matches':
                lines.append('These calculations match the printed values; accounting scope is unspecified.')
            else:raise ValueError('Unknown source reconciliation status cannot be rendered.')
        else:raise ValueError('Unknown canonical claim cannot be rendered.')
        source_inputs=value.get('source_inputs',())
        if any(source['unit']=='billion' for source in source_inputs):
            roles={'reported_actual':'actual','broker_estimate':'broker estimate'}
            originals=[]
            for source in source_inputs:
                role=roles.get(source['kind'],source['kind'])
                if source['kind']=='reported_actual' and source['period']!=value['period']:
                    role='prior actual'
                originals.append(' '.join(str(part) for part in
                    (source['period'],role,source['currency'],source['value_text'],source['unit'])
                    if part is not None))
            lines.append(prefix+'original source inputs: '+'; '.join(originals)+'.')
    if shared_context:
        return context+':\n\n'+'\n'.join('- '+line for line in lines)+'\n\n'+_source_list(answer)
    return '\n\n'.join((*lines,_source_list(answer)))


def run_chat(config,mode='live',input_fn=input,output_fn=print,*,diagnostics=False):
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
        company_count=len(coverage['companies'])
        noun='company' if company_count==1 else 'companies'
        banner=_coverage_banner(coverage,inference) if diagnostics else (
            f"Reports available for {company_count} {noun}. Ask about a company, period, or metric.",)
        for line in banner:
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
            try:output_fn(render_answer(agent.answer(question),diagnostics=diagnostics))
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
    parser.add_argument('--diagnostics',action='store_true',help='Show detailed execution and source labels.')
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
            print(json.dumps(_json_value(result),indent=2,ensure_ascii=False) if args.format=='json' else render_answer(result,diagnostics=args.diagnostics))
            return 0 if result.status=='answered' else 1
        else:
            return run_chat(config,mode,diagnostics=args.diagnostics)
        print(json.dumps(_json_value(result),indent=2,ensure_ascii=False))
        return 0
    except KeyboardInterrupt:return 0
    except Exception:
        print('Corpus operation failed. Check private configuration, reviewed source files, SQLite integrity and provider availability.')
        return 2


if __name__=='__main__':raise SystemExit(main())
