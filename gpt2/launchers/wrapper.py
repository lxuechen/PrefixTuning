import os
import platform
import random
import subprocess
import sys
import uuid


def mynlprun_wrapper(
    command,
    mynlprun=True,
    logs_prefix="/nlp/scr/lxuechen/logs",
    priority="standard",
    train_dir=None,
    job_name=None,
    salt_length=8,
    conda_env="lxuechen-torch",
    memory="16g",
):
    if mynlprun:
        if train_dir is not None:
            log_path = f"{train_dir}/log.out"
        else:
            log_path = f"{logs_prefix}/{create_random_job_id()}.out"
        # Don't need to exclude jagupard[4-8] per https://stanfordnlp.slack.com/archives/C0FSH01PY/p1621469284003100
        # TODO: Don't remember why I'm excluding `jagupard14`
        wrapped_command = (
            f"nlprun -x=john0,john1,john2,john3,john4,john5,john6,john7,john8,john9,john10,john11,"
            f"jagupard14 "
            f"-a {conda_env} "
            f"-o {log_path} "
            f"-p {priority} "
            f"--memory {memory} "
        )
        if job_name is not None:
            # Suffix with uid just in case you get a job name collision!
            this_id = uuid.uuid4().hex[:salt_length]
            job_name = f"{job_name}-{this_id}"
            wrapped_command += f"--job_name {job_name} "
        wrapped_command += f"'{command}'"

        if train_dir is not None:
            # First mkdir, then execute the command.
            wrapped_command = f'mkdir -p "{train_dir}"\n' + wrapped_command
    else:
        wrapped_command = command
    return wrapped_command


# Shameless copy from https://github.com/stanfordnlp/cluster/blob/main/slurm/nlprun.py
def create_random_job_id():
    # handle Python 2 vs. Python 3
    if sys.version_info[0] < 3:
        return subprocess.check_output("whoami")[:-1] + "-job-" + str(random.randint(0, 5000000))
    else:
        return str(subprocess.check_output("whoami")[:-1], encoding="utf8") + "-job-" + str(random.randint(0, 5000000))


def report_node_and_scratch_dirs():
    machine_name = platform.node().split(".")[0]
    scratch_dirs = os.listdir(f"/{machine_name}")
    return machine_name, scratch_dirs


# These are useful for naming directories with float or int parameter values.
def float2str(x, precision=8):
    return f"{x:.{precision}f}".replace('.', "_")


def int2str(x, leading_zeros=8):
    return f"{x:{leading_zeros}0d}"
