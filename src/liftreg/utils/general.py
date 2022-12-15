import os
from glob import glob
import subprocess

def make_dir(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def get_class(kls):
    parts = kls.split('.')
    module = ".".join(parts[:-1])
    m = __import__(module)
    for comp in parts[1:]:
        m = getattr(m, comp)
    return m

def get_git_revisions_hash():
    hashes = []
    hashes.append(subprocess.check_output(['git', 'rev-parse', 'HEAD']).strip().decode('ascii'))
    return hashes