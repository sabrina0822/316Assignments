import socket 
import argparse
from re import match

'''
python DnsClient [-t timeout] [-r max-retries] [-p port] [-mx | nx] @server name

-t (5): gives how long to wait, in seconds before retransmitting an unanswered query. 
-r (3): max # of times to retransmit an unanswered query before giving up
-p (53): UDP port number of DNS server 
-mx (IP address): send a mail server query 
-ns (IP address): send a name server query 

'''

def valid_server_address(server_address):
    try:
        if match(r'@*.*.*.*', server_address): # TODO add better regex for valid IP address if wanted
            return server_address
        else:
            raise ValueError
    except:
        msg = "not a valid server address: {0!r}".format(server_address)
        raise argparse.ArgumentTypeError(msg)

def collect_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', type=float, default=5.0)
    parser.add_argument('-r', type=int, default=3)
    parser.add_argument('-p', type=int, default=53)
    group = parser.add_mutually_exclusive_group()
    group.add_argument('-ns', action='store_true', default=False)
    group.add_argument('-mx', action='store_true', default=False) 
    parser.add_argument('server', type=valid_server_address)
    parser.add_argument('name', type=str)

    return  parser.parse_args()

def querry_server(ip, port, timeout, retries, packet):
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.sendto(packet, (ip, port))
    sock.settimeout(timeout)
    while retries > 0:
        try:
            data, addr = sock.recvfrom(1024)
            print(f"received {data} from {addr}")
        except socket.timeout:
            retries -= 1
            print("timeout")
            return None
    

if __name__ == "__main__":
    args = collect_args()
    
    timeout = args.t
    retries = args.r
    port = args.p
    mail_server = args.mx
    name_server = args.ns
    ip_address = args.server
    domain_name = args.name

    print(f"timeout {timeout}\nretries {retries}\nport {port}\nmail_server {mail_server}\nname_server {name_server}\nserver {ip_address}\nname {domain_name}")



