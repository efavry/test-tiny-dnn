import subprocess

trainer_cmd = ['./bin/test_release',
               '--learning_rate', '1',
               '--epochs', '30',
               '--minibatch_size', '16',
               '--backend_type', 'internal']

proc = subprocess.Popen(trainer_cmd, stdin=subprocess.PIPE,
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   universal_newlines=True)
print('subprocess started')
with open('sample_input', 'r') as input_file:
    for l in input_file:
        proc.stdin.write(l)
        # from_stdout, from_stderr = proc.communicate(l)
        # print('Sent ', l, ' down the pipe')

# proc.wait()
# print('Stdout of the process')
# for l in proc.stdout:
    # print(l)

from_stdout, from_stderr = proc.communicate()
print('Stdout of the process')
for l in from_stdout:
    print(l, end='')

print('stderr of the process')
for l in from_stderr:
    print(l, end='')

print('Return code : ', proc.returncode)
