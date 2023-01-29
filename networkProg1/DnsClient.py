import secrets
import socket 
import argparse
import random
from re import match
from time import time_ns

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
    start_time = time_ns()
    sock.sendto(packet, (ip, port))
    sock.settimeout(timeout)
    count = 0
    while retries - count > 0:
        try:
            data, _ = sock.recvfrom(1024)
            print("data", data)
            end_time = time_ns()
            print(f"Response received after {end_time - start_time} seconds ({count} retries)")
            return data, (end_time - start_time)
        except socket.timeout:
            count += 1
            print("timeout") # TODO remove
            return None, None
    else:
        print(f"ERROR\tMaximum number of retries {retries} exceeded")
        return None, None
    
def random_id():
    return (secrets.token_bytes(2))
#converts to hex, then removes first two numbers to get rid of 0x
def convert_to_hex(id):
    return hex(id)[2:].zfill(4) #zfill pads with 0s to make 16 bits
    
def create_header(id): 
    #headers consist of a 16 bit id, 16 bit flags, 16 bit question count, 16 bit answer count, 16 bit authority count, 16 bit additional count
    #in a flag, | QR | OPCODE (0) | AA (0)| TC (0)| RD | RA | Z | RCODE |
    array = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    #print ((bin(packet_id)[2:].zfill(16)) + '0000000100000000'+'0000000000000001'+'0000000000000000'+'0000000000000000'+'0000000000000000' )
    #header = int((bin(packet_id)[2:].zfill(16)) + '0000000100000000'+'0000000000000001'+'0000000000000000'+'0000000000000000'+'0000000000000000', 2)
    #header = hex(header)
    return (id + bytes(array))


#parameters, domain name, qtype (hex number representing type of query)
def create_question(name, QTYPE):
    #QNAME is a domain name, sequence of lables where each label begins with a length octet followed by that number of octets
    #the domain name terminates with the zero length octect (null label of the root)
    QNAME = []
    for i in name.split("."):
        length = len(i)
        ascii = i.encode()
        QNAME += [length] + list(ascii)
    #QTYPE, 16 bit code specifying the type of query
        # 0x0001 = A
        # 0x0002 = NS
        # 0x000f = MX   
    #QCLASS, 16 bit code specifying the class of query (always use 0x0001)
    QCLASS = 1

    question = QNAME + [0, 1, QTYPE, QCLASS]
    return bytes(question)

def parse_packet_header(packet_data):
    # Assuming that packet data is in bits
    # create dictionary to store packet header fields in BITS
    packet_header_fields = {}

    # id is the first 16 bits of the packet in the header
    # Take first 4 hex characters and convert to bin
    packet_header_fields['id'] = '0b' + packet_data[0:16]

    # flags are the next 16 bits of the packet in the header
    # | QR | OPCODE | AA | TC | RD | RA | Z | RCODE |
    packet_header_fields['qr'] = '0b' + packet_data[16:17]
    packet_header_fields['opcode'] = '0b' + packet_data[17:21]
    packet_header_fields['aa'] = '0b' + packet_data[21:22]
    packet_header_fields['tc'] = '0b' + packet_data[22:23]
    packet_header_fields['rd'] = '0b' + packet_data[23:24]
    packet_header_fields['ra'] = '0b' + packet_data[24:25]
    packet_header_fields['z'] = '0b' + packet_data[25:28]
    packet_header_fields['rcode'] = '0b' + packet_data[28:32]

    # QDCOUNT is the next 16 bits specifying the number entries in the question section
    packet_header_fields['qdcount'] = '0b' + packet_data[32:48]

    # ANCOUNT is the next 16 bits specifying the number of resource records in the answer section
    packet_header_fields['ancount'] = '0b' + packet_data[48:64]
    
    # NSCOUNT is the next 16 bits specifying the number of name server resource records in the authority records section
    packet_header_fields['nscount'] = '0b' + packet_data[64:80]

    # ARCOUNT is the next 16 bits specifying the number of resource records in the additional records section
    packet_header_fields['arcount'] = '0b' + packet_data[80:96]

    return packet_header_fields

def parse_packet_questions(packet_data):
    # Assuming that packet data is in bits
    # create dictionary to store packet questions in BITS
    # question begins at the 12th octect (96th bit)
    packet_question_fields = {}

    # QNAME is a domain name represented by a sequence of labels, where each label 
    # begins with a length octet followed by that number of octets. 
    # The domain name terminates with the zero-length octet, representing the null label of the root
    octet = 12
    qname = '0b'
    while ('0b' + packet_data[octet*8, (octet+1)*8] != '0b00000000'):
        qname += packet_data[octet*8, (octet+1)*8]
        octet += 1
    else:
        octet += 1

    packet_question_fields['qname'] = '0b' + qname

    # QTYPE is the next 16 bits specifying the type of query
    packet_question_fields['qtype'] = '0b' + packet_data[octet*8:(octet+1)*8]

    # QCLASS is the next 16 bits specifying the class of query
    octet += 1
    packet_question_fields['qclass'] = '0b' + packet_data[octet*8, (octet+1)*8]

    return packet_question_fields, octet+1

def parse_packet_answers(packet_data, starting_octet):
    # Assuming that packet data is in bits
    # create dictionary to store packet answers in BITS
    # Traverse packet data until we reach the answer section that starts with 11
    packet_answer_fields = {}

    if ('0b' + packet_data[starting_octet*8:starting_octet*8+2] != '0b11'):
        print("No offset")

    octect_offset = int('0b' + packet_data[starting_octet*8+2:(starting_octet+1)*8], 2)

    current_octet = octect_offset
    name = '0b'
    while ('0b' + packet_data[current_octet*8, (current_octet+1)*8] != '0b00000000'):
        name += packet_data[current_octet*8, (current_octet+1)*8]
        current_octet += 1
    else:
        current_octet += 1
    
    # NAME use offset to find name from question section
    packet_answer_fields['name'] = name

    # TYPE is the next 16 bits specifying the type of query
    packet_answer_fields['type'] = '0b' + packet_data[current_octet*8:(current_octet+1)*8]

    # CLASS is the next 16 bits specifying the class of query
    current_octet += 1
    packet_answer_fields['class'] = '0b' + packet_data[current_octet*8, (current_octet+1)*8]

    # TTL is the next 32 bits specifying the time to live
    current_octet += 1
    packet_answer_fields['ttl'] = '0b' + packet_data[current_octet*8, (current_octet+2)*8]

    # RDLENGTH is the next 16 bits specifying the length of the RDATA field
    current_octet += 2
    packet_answer_fields['rdlength'] = '0b' + packet_data[current_octet*8, (current_octet+1)*8]

    # RDATA is the next RDLENTH bits specifying the data
    current_octet += 1
    packet_answer_fields['rdata'] = '0b' + packet_data[current_octet*8, (current_octet+int(packet_answer_fields['rdlength'], 2))*8]

    return packet_answer_fields

def print_answer(num_answers, type, alias, IP_address, pref, seconds, auth):
    print(f"***Answer Section ({num_answers} records)***")
    type = int(type, 2)
    type = hex(type)
    match type:
        case '0x0001':
            # IP <tab> [ip address] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("IP\t" + IP_address + "\t" + seconds + "\t" + auth)
        case '0x002':
            # NS <tab> [alias] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("NS\t" + alias + "\t" + seconds + "\t" + auth)
        case '0x0005':
            # CNAME <tab> [alias] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("CNAME\t" + alias + "\t" + seconds + "\t" + auth)
        case '0x000f':
            # MX <tab> [alias] <tab> [pref] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("MX\t" + alias + "\t" + pref + "\t" + seconds + "\t" + auth)

def print_additional(num_records):
    print(f"***Additional Section ({num_records} records)***")
    
    if num_records == 0:
        print("NOTFOUND")
    
    for i in range(num_records):
            print("records") # TODO
 

def read_packet(packet, id):

    packet_header_fields = parse_packet_header(packet)
    packet_question_fields, current_octet = parse_packet_questions(packet)
    packet_answer_fields = parse_packet_answers(packet, current_octet)

    # check if id matches
    if packet_header_fields['id'] != id:
        print("ERROR\tID mismatch")
        return
    
    # check if response is 1
    if packet_header_fields['qr'] != '0b1':
        print("ERROR\tNot a response")
        return
    
    # Check if server supports recursion
    if packet_header_fields['rd'] != '0b1':
        print("ERROR\tRecursion not supported")
        return

    # Check if response is valid
    match packet_header_fields['rcode']:
        case '0b0000':
            print("OK") # TODO remove
        case '0b0001':
            print("ERROR\tFormat error: the name server was unable to interpret the query")
        case '0b0010':
            print("ERROR\tServer failure: the name server was unable to process this query due to a problem with the name server")
        case '0b0011':
            print("NOTFOUND")
        case '0b0100':
            print("ERROR\tNot Implemented: the name server does not support the requested kind of query")
        case '0b0101':
            print("ERROR\tRefused: the name server refuses to perform the specified operation for policy reasons")
        case _:
            print("ERROR\tUnknown error")
    
    

    
def qtype(mail_server, name_server):
    if mail_server is True: 
        return 0x000f
    elif name_server is True:
        return 0x0002
    else: 
        return 0x0001

if __name__ == "__main__":
    args = collect_args()
    
    timeout = args.t
    retries = args.r
    port = args.p
    print('port: ', port)
    mail_server = args.mx
    name_server = args.ns
    ip_address = args.server[1:]
    print(ip_address)
    domain_name = args.name
    id = random_id()

    #header section 
    header = create_header(id)
    print (header)

    #question section
    server_type = qtype(mail_server, name_server)
    question_packet = header + create_question(domain_name, server_type)
    print('question packet')
    print(question_packet)
    response_packet, time = querry_server(ip_address, port, timeout, retries, question_packet)

    read_packet(response_packet, id)
    

    print(f"timeout {timeout}\nretries {retries}\nport {port}\nmail_server {mail_server}\nname_server {name_server}\nserver {ip_address}\nname {domain_name}")



