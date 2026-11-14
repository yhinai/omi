"""Explicit enrollment from a deliberate local sample; never a second device reader."""
import argparse
import os
from pathlib import Path
from config import ROOT
from speakers import PROFILE


def main(argv=None):
    p=argparse.ArgumentParser(description='Manage your local voice profile; no automatic recording.')
    sub=p.add_subparsers(dest='action',required=True)
    enroll=sub.add_parser('enroll',help='Enroll from 9–30 seconds of only your voice')
    enroll.add_argument('--wav',type=Path,required=True)
    sub.add_parser('status')
    sub.add_parser('forget',help='Delete the local voice embedding; leave recordings untouched')
    args=p.parse_args(argv)
    if args.action=='status':print('Voice profile: '+('enrolled' if PROFILE.is_file() else 'not enrolled'));return
    if args.action=='forget':PROFILE.unlink(missing_ok=True);print('Local voice profile deleted.');return
    print('Creating a local voice embedding from your supplied sample. Raw audio is not copied.',flush=True)
    executable=str(ROOT/'.venv-speaker/bin/python')
    os.execv(executable,[executable,str(ROOT/'voice_worker.py'),'--enroll-wav',str(args.wav.resolve())])

if __name__=='__main__':main()
