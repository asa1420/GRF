import paramiko
# Connect to remote host
client = paramiko.SSHClient()
client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
client.connect('athens.ee.ucl.ac.uk', username='zceesaa', password='WorkHarder8244')
# Setup sftp connection and transmit this script
sftp = client.open_sftp()
sftp.put(__file__, '/tmp/SimplerModel.py')
sftp.close()
# Run the transmitted script remotely without args and show its output.
# SSHClient.exec_command() returns the tuple (stdin,stdout,stderr)
stdout = client.exec_command('python /tmp/SimplerModel.py')[1]
print("success")
for line in stdout:
    # Process each line in the remote output
    print (line)
client.close()
sys.exit(0)