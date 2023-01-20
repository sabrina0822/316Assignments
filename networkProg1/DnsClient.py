import socket 
import argparse

'''
python DnsClient [-t timeout] [-r max-retries] [-p port] [-mx | nx] @server name

-t (5): gives how long to wait, in seconds before retransmitting an unanswered query. 
-r (3): max # of times to retransmit an unanswered query before giving up
-p (53): UDP port number of DNS server 
-mx (IP address): send a mail server query 
-ns (IP address): send a name server query 

'''
parser = argparse.ArgumentParser()
parser.add_argument('-t', type=float, default=5.0)
parser.add_argument('-r', type=int, default=3)
parser.add_argument('-p', type=int, default=53)

group = parser.add_mutually_exclusive_group()
group.add_argument('-ns', action='store_true', default=False)
group.add_argument('-mx', action='store_true', default=False) 


args = parser.parse_args()


timeout = args.t
retries = args.r
port = args.p
mail_server = args.mx
name_server = args.ns

print(f"timeout {timeout}\nretries {retries}\nport {port}\nmail_server {mail_server}\nname_server {name_server}")