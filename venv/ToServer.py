import paramiko
# Connect to remote host
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
print('hola')
client.connect(hostname='athens.ee.ucl.ac.uk', username='zceesaa', password='WorkHarder8244')
# Setup sftp connection and transmit this script
print("ayo")
sftp = client.open_sftp()
sftp.put(__file__, '/tmp/train.py')
sftp.close()
print("hi")
# Run the transmitted script remotely without args and show its output.
# SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
stdin, stdout, stderr = client.exec_command('python /tmp/train.py')[1]
stdout = stdout.readlines()
print(stdout)
print("success")
for line in stdout:
    # Process each line in the remote output
    print (line)
client.close()
sys.exit(0)