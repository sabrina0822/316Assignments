import socket 
import argparse
import random
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
    
def random_id():
    return random.getrandbits(16) #16 bit id in decimal

#converts to binary, then removes first two numbers to get rid of 0b
def convert_to_bits(id):
    return bin(id)[2:].zfill(16) #zfill pads with 0s to make 16 bits

#converts to hex, then removes first two numbers to get rid of 0x
def convert_to_hex(id):
    return hex(id)[2:].zfill(4) #zfill pads with 0s to make 16 bits
    
def create_header(id): 
    #headers consist of a 16 bit id, 16 bit flags, 16 bit question count, 16 bit answer count, 16 bit authority count, 16 bit additional count
    #in a flag, | QR | OPCODE (0) | AA (0)| TC (0)| RD | RA | Z | RCODE |
    header = convert_to_bits(id)+"0000000100000000"+"0000000000000001"+"0000000000000000"+"0000000000000000"+"0000000000000000"
    return header

#parameters, domain name, qtype (hex number representing type of query)
def create_question(name, QTYPE):
    #QNAME is a domain name, sequence of lables where each label begins with a length octet followed by that number of octets
    #the domain name terminates with the zero length octect (null label of the root)
    QNAME = ""
    for i in name.split("."):
        length = convert_to_hex(len(i))
        ascii = i.encode().hex()
        QNAME += length + ascii
    #QTYPE, 16 bit code specifying the type of query
        # 0x0001 = A
        # 0x0002 = NS
        # 0x000f = MX   
    #QCLASS, 16 bit code specifying the class of query (always use 0x0001)
    QCLASS = "0000000000000001"

    return QNAME+"00"+QTYPE+QCLASS
    
    

def parse_packet_header(packet_data):
    # Assuming that packet data is in hex
    # create dictionary to store packet header fields in BITS
    packet_header_fields = {}

    # id is the first 16 bits of the packet in the header
    # Take first 4 hex characters and convert to bin
    packet_header_fields['id'] = bin(packet_data[0:4])

    # flags are the next 16 bits of the packet in the header
    # | QR | OPCODE | AA | TC | RD | RA | Z | RCODE |
    flag_bits = bin(packet_data[4:8])
    packet_header_fields['qr'] = flag_bits[0:1]
    packet_header_fields['opcode'] = flag_bits[1:5]
    packet_header_fields['aa'] = flag_bits[5:6]
    packet_header_fields['tc'] = packet_data[6:7]
    packet_header_fields['rd'] = packet_data[7:8]
    packet_header_fields['ra'] = packet_data[8:9]
    packet_header_fields['z'] = packet_data[9:12]
    packet_header_fields['rcode'] = packet_data[12:16]

    # QDCOUNT is the next 16 bits specifying the number entries in the question section
    packet_header_fields['qdcount'] = bin(packet_data[8:12])

    # ANCOUNT is the next 16 bits specifying the number of resource records in the answer section
    packet_header_fields['ancount'] = bin(packet_data[12:16])
    
    # NSCOUNT is the next 16 bits specifying the number of name server resource records in the authority records section
    packet_header_fields['nscount'] = bin(packet_data[16:20])

    # ARCOUNT is the next 16 bits specifying the number of resource records in the additional records section
    packet_header_fields['arcount'] = bin(packet_data[20:24])

    return packet_header_fields

def parse_packet_question(packet_data):
    # Assuming that packet data is in hex
    # create dictionary to store packet question fields in BITS
    packet_question_fields = {}

    # Question section starts at 25th hex character (index 24)
    
    
    
    return packet_question_fields


def read_packet(packet_header_fields, id):
    
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

    packet_id = random_id()
    
    question_packet = create_packet()

    response_packet = querry_server(ip_address, port, timeout, retries, question_packet)

    read_packet(response_packet, packet_id)
    

    print(f"timeout {timeout}\nretries {retries}\nport {port}\nmail_server {mail_server}\nname_server {name_server}\nserver {ip_address}\nname {domain_name}")



