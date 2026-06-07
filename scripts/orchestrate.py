#!/usr/bin/env python3
"""
Orchestration script - monitors SLURM jobs and submits next steps automatically.
Run this from the LOCAL machine via SSH polling.
It writes state to /work/jl1401/icl_protein_disease/logs/orchestrate_state.json
"""
import subprocess, json, os, sys, time
from pathlib import Path

ROOT = '/work/jl1401/icl_protein_disease'
STATE_FILE = f'{ROOT}/logs/orchestrate_state.json'

def run(cmd):
    r = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    return r.stdout.strip(), r.stderr.strip(), r.returncode

def sbatch(script):
    out, err, rc = run(f'cd {ROOT} && sbatch {script}')
    if rc == 0:
        jid = out.split()[-1]
        print(f'  Submitted {script} -> job {jid}', flush=True)
        return jid
    else:
        print(f'  ERROR submitting {script}: {err}', flush=True)
        return None

def job_status(jid):
    """Returns: PENDING, RUNNING, COMPLETED, FAILED, UNKNOWN"""
    out, _, rc = run(f'squeue -j {jid} -h -o %T 2>/dev/null')
    if rc == 0 and out:
        return out.strip()
    # job no longer in queue - check sacct
    out, _, _ = run(f'sacct -j {jid} --format=State --noheader 2>/dev/null | head -1')
    state = out.strip().split()[0] if out.strip() else 'UNKNOWN'
    return state

def all_done(job_ids, state):
    """Check if all jobs in list are COMPLETED."""
    for jid in job_ids:
        s = job_status(jid)
        state['job_statuses'][jid] = s
        if s not in ('COMPLETED',):
            return False
    return True

def any_failed(job_ids, state):
    for jid in job_ids:
        s = job_status(jid)
        state['job_statuses'][jid] = s
        if s.startswith('CANCELLED') or s in ('FAILED', 'TIMEOUT', 'OUT_OF_MEMORY'):
            return jid
    return None

def result_exists(path):
    out, _, rc = run(f'test -f {ROOT}/{path} && echo yes')
    return out.strip() == 'yes'

def load_state():
    out, _, rc = run(f'cat {STATE_FILE} 2>/dev/null')
    if rc == 0 and out:
        try:
            return json.loads(out)
        except:
            pass
    return {'phase': 'init', 'jobs': {}, 'job_statuses': {}, 'errors': []}

def save_state(state):
    run(f"echo '{json.dumps(state, indent=2)}' > {STATE_FILE}")

def check_log_for_error(log_glob):
    out, _, _ = run(f'ls {ROOT}/logs/{log_glob} 2>/dev/null | tail -1')
    if not out:
        return None
    log_file = out.strip()
    out, _, _ = run(f'grep -i "error\\|traceback\\|exception" {log_file} | tail -5')
    return out if out else None

def main():
    state = load_state()
    print(f"\n{'='*60}", flush=True)
    print(f"Orchestrator check - phase: {state['phase']}", flush=True)
    print(f"{'='*60}", flush=True)

    # ---- PHASE: INIT - submit preprocess jobs ----
    if state['phase'] == 'init':
        print("Phase: Submitting preprocess jobs (C and G blocks)...", flush=True)
        jid_c = sbatch('slurm_jobs/preprocess_c.sh')
        jid_g = sbatch('slurm_jobs/preprocess_g.sh')
        # Also submit I model evals immediately (data already exists)
        jid_eval_ic = sbatch('slurm_jobs/eval_i_on_c.sh')
        jid_eval_ig = sbatch('slurm_jobs/eval_i_on_g.sh')
        jid_dnn_i   = sbatch('slurm_jobs/baseline_dnn_i.sh')
        state['jobs']['preprocess_c'] = jid_c
        state['jobs']['preprocess_g'] = jid_g
        state['jobs']['eval_i_on_c']  = jid_eval_ic
        state['jobs']['eval_i_on_g']  = jid_eval_ig
        state['jobs']['baseline_dnn_i'] = jid_dnn_i
        state['phase'] = 'preprocessing'
        save_state(state)
        return

    # ---- PHASE: PREPROCESSING ----
    if state['phase'] == 'preprocessing':
        jids = [j for j in [state['jobs'].get('preprocess_c'), state['jobs'].get('preprocess_g')] if j]
        failed = any_failed(jids, state)
        if failed:
            log_err = check_log_for_error(f'preprocess_*_{failed}*')
            state['errors'].append({'phase': 'preprocessing', 'job': failed, 'log': log_err})
            state['phase'] = 'error'
            save_state(state)
            print(f"ERROR: preprocess job {failed} failed. Log snippet:\n{log_err}", flush=True)
            return
        if all_done(jids, state):
            print("Preprocess done! Submitting training jobs...", flush=True)
            jid_tc = sbatch('slurm_jobs/train_c_model.sh')
            jid_tg = sbatch('slurm_jobs/train_g_model.sh')
            # Also submit C/G baselines (data ready)
            jid_xgb_c = sbatch('slurm_jobs/baseline_xgboost_c.sh')
            jid_xgb_g = sbatch('slurm_jobs/baseline_xgboost_g.sh')
            jid_dnn_c  = sbatch('slurm_jobs/baseline_dnn_c.sh')
            jid_dnn_g  = sbatch('slurm_jobs/baseline_dnn_g.sh')
            state['jobs'].update({
                'train_c': jid_tc, 'train_g': jid_tg,
                'baseline_xgb_c': jid_xgb_c, 'baseline_xgb_g': jid_xgb_g,
                'baseline_dnn_c': jid_dnn_c, 'baseline_dnn_g': jid_dnn_g
            })
            state['phase'] = 'training'
            save_state(state)
        else:
            print(f"Preprocess still running. Job statuses: {state['job_statuses']}", flush=True)
        return

    # ---- PHASE: TRAINING ----
    if state['phase'] == 'training':
        jids = [j for j in [state['jobs'].get('train_c'), state['jobs'].get('train_g')] if j]
        failed = any_failed(jids, state)
        if failed:
            log_err = check_log_for_error(f'train_*_{failed}*')
            state['errors'].append({'phase': 'training', 'job': failed, 'log': log_err})
            state['phase'] = 'error'
            save_state(state)
            print(f"ERROR: training job {failed} failed. Log snippet:\n{log_err}", flush=True)
            return
        if all_done(jids, state):
            print("Training done! Submitting cross-block eval jobs...", flush=True)
            jobs = {}
            for name, script in [
                ('eval_c_on_c', 'slurm_jobs/eval_c_on_c.sh'),
                ('eval_c_on_i', 'slurm_jobs/eval_c_on_i.sh'),
                ('eval_c_on_g', 'slurm_jobs/eval_c_on_g.sh'),
                ('eval_g_on_g', 'slurm_jobs/eval_g_on_g.sh'),
                ('eval_g_on_i', 'slurm_jobs/eval_g_on_i.sh'),
                ('eval_g_on_c', 'slurm_jobs/eval_g_on_c.sh'),
            ]:
                jobs[name] = sbatch(script)
            state['jobs'].update(jobs)
            state['phase'] = 'evaluating'
            save_state(state)
        else:
            print(f"Training still running. Job statuses: {state['job_statuses']}", flush=True)
        return

    # ---- PHASE: EVALUATING ----
    if state['phase'] == 'evaluating':
        # Submit i_half eval if not already submitted
        if not state['jobs'].get('eval_i_half_on_i'):
            jid = sbatch('slurm_jobs/eval_i_half_on_i.sh')
            if jid:
                state['jobs']['eval_i_half_on_i'] = jid
                save_state(state)

        eval_keys = ['eval_c_on_c','eval_c_on_i','eval_c_on_g',
                     'eval_g_on_g','eval_g_on_i','eval_g_on_c',
                     'eval_i_on_c','eval_i_on_g','eval_i_half_on_i']
        jids = [state['jobs'].get(k) for k in eval_keys if state['jobs'].get(k)]
        failed = any_failed(jids, state)
        if failed:
            log_err = check_log_for_error(f'eval_*_{failed}*')
            state['errors'].append({'phase': 'evaluating', 'job': failed, 'log': log_err})
            print(f"ERROR: eval job {failed} failed. Will attempt recovery.", flush=True)
            # mark error but don't stop - other evals may still complete
            state['phase'] = 'error'
            save_state(state)
            return
        if all_done(jids, state):
            print("All cross-block evals done!", flush=True)
            state['phase'] = 'done'
            save_state(state)
            print("\n=== ALL EXPERIMENTS COMPLETE ===", flush=True)
            # Print summary
            for result_file in ['i_model_on_c','i_model_on_g','c_model_on_c',
                                 'c_model_on_i','c_model_on_g','g_model_on_g',
                                 'g_model_on_i','g_model_on_c','i_half_model_on_i']:
                out, _, _ = run(f"python3 -c \"import json; d=json.load(open('{ROOT}/results/{result_file}.json')); print('{result_file}:', d.get('summary',{{}}).get('mean_auroc','N/A'))\" 2>/dev/null")
                if out:
                    print(f"  {out}", flush=True)
        else:
            print(f"Evals still running. Job statuses: {state['job_statuses']}", flush=True)
        return

    # ---- PHASE: ERROR ----
    if state['phase'] == 'error':
        print(f"In error state. Recent errors:", flush=True)
        for e in state['errors'][-3:]:
            print(f"  {e}", flush=True)
        print("Manual intervention needed. Fix the error and reset phase in state file.", flush=True)
        return

    # ---- PHASE: DONE ----
    if state['phase'] == 'done':
        print("All experiments completed!", flush=True)
        return

if __name__ == '__main__':
    main()
