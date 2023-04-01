import secrets
import socket 
import argparse
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
        if match(r'@*.*.*.*', server_address):
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

def query_server(ip, port, timeout, retries, packet): 
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.settimeout(timeout)
    count = 0
    start_time = time_ns()
    while retries - count > 0:
        try:
            
            sock.sendto(packet, (ip, port))
            data, addr = sock.recvfrom(1024)
            end_time = time_ns()
            print(f"Response received after {(end_time - start_time)/1000000000} seconds ({count} retries)")
            return data, (end_time - start_time)
        except socket.timeout:
            count += 1
    else:
        print(f"ERROR\tMaximum number of retries {retries} exceeded")

    sock.close()
    return None, None
    
def random_id():
    id = (secrets.token_bytes(2))
    return id

def convert_bytes_to_bin(num):
    num = bytes([1]) + num
    return bin(int(num.hex(), 16))[3:]
    
def create_header(id):
    #headers consist of a 16 bit id, 16 bit flags, 16 bit question count, 16 bit answer count, 16 bit authority count, 16 bit additional count
    #in a flag, | QR | OPCODE (0) | AA (0)| TC (0)| RD | RA | Z | RCODE |
    array = [1, 0, 0, 1, 0, 0, 0, 0, 0, 0]
    return (id+ (bytes(array)))


#parameters, domain name, qtype (hex number representing type of query)
def create_question(name, QTYPE):
    #QNAME is a domain name, sequence of lables where each label begins with a length octet followed by that number of octets
    #the domain name terminates with the zero length octect (null label of the root)
    QNAME = []
    for i in name.split("."):
        length = len(i)
        ascii = i.encode()
        QNAME += [length] + list(ascii)
    #QCLASS, 16 bit code specifying the class of query (always use 0x0001)
    QCLASS = [0, 1]

    question = QNAME + [0] + QTYPE + QCLASS
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
    while (('0b' + packet_data[octet*8:(octet+1)*8]) != '0b00000000'):
        qname += packet_data[octet*8:(octet+1)*8]
        octet += 1
    else:
        octet += 1

    packet_question_fields['qname'] = qname

    # QTYPE is the next 16 bits specifying the type of query
    packet_question_fields['qtype'] = '0b' + packet_data[octet*8:(octet+2)*8]

    # QCLASS is the next 16 bits specifying the class of query
    octet += 2
    packet_question_fields['qclass'] = '0b' + packet_data[octet*8:(octet+2)*8]

    return packet_question_fields, octet+2

def parse_packet_records(packet_data, starting_octet):
    # Assuming that packet data is in bits
    # create dictionary to store packet answers in BITS
    # Traverse packet data until we reach the answer section that starts with 11
    packet_record_fields = {}


    current_octet = starting_octet
    name = '0b'
    num_offsets = 0
    while ('0b' + packet_data[current_octet*8:(current_octet+1)*8] != '0b00000000'):
        if ('0b' + packet_data[current_octet*8:current_octet*8+2] == '0b11'):
            if num_offsets == 0:
                offset_octet = current_octet
            current_octet = int('0b' + packet_data[current_octet*8+2:(current_octet+2)*8], 2)
            num_offsets += 1
        else:
            name += packet_data[current_octet*8:(current_octet+1)*8]
            current_octet += 1

    else:
        if num_offsets > 0:
            current_octet = offset_octet + 2
        else:
            current_octet += 1
    
    # NAME use offset to find name from question section
    packet_record_fields['name'] = name

    # TYPE is the next 16 bits specifying the type of query
    packet_record_fields['type'] = '0b' + packet_data[current_octet*8:(current_octet+2)*8]

    # CLASS is the next 16 bits specifying the class of query
    current_octet += 2
    packet_record_fields['class'] = '0b' + packet_data[current_octet*8:(current_octet+2)*8]

    # TTL is the next 32 bits specifying the time to live
    current_octet += 2
    packet_record_fields['ttl'] = '0b' + packet_data[current_octet*8:(current_octet+4)*8]

    # RDLENGTH is the next 16 bits specifying the length of the RDATA field
    current_octet += 4
    packet_record_fields['rdlength'] = '0b' + packet_data[current_octet*8:(current_octet+2)*8]

    # RDATA is the next RDLENTH bits specifying the data
    octet_before_rdata = current_octet + 2
    current_octet += 2
    type = int(packet_record_fields['type'], 2)
    type = hex(type)

    match type:
        case '0x1': #IP
            packet_record_fields['rdata'] = '0b' + packet_data[current_octet*8:(current_octet+4)*8]
            current_octet += 4
        case '0x2': #NS
            ns = '0b'
            num_offsets = 0
            while ('0b' + packet_data[current_octet*8:(current_octet+1)*8] != '0b00000000'):
                if ('0b' + packet_data[current_octet*8:current_octet*8+2] == '0b11'):
                    if num_offsets == 0:
                        offset_octet = current_octet
                    current_octet = int('0b' + packet_data[current_octet*8+2:(current_octet+2)*8], 2)
                    num_offsets += 1
                else:
                    ns += packet_data[current_octet*8:(current_octet+1)*8]
                    current_octet += 1
            else:
                if num_offsets > 0:
                    current_octet = offset_octet + 2
                else:
                    current_octet += 1

            packet_record_fields['rdata'] = ns
        
        case '0x5': #CNAME
            cname = '0b'
            num_offsets = 0
            while ('0b' + packet_data[current_octet*8:(current_octet+1)*8] != '0b00000000'):
                if ('0b' + packet_data[current_octet*8:current_octet*8+2] == '0b11'):
                    if num_offsets == 0:
                        offset_octet = current_octet
                    current_octet = int('0b' + packet_data[current_octet*8+2:(current_octet+2)*8], 2)
                    num_offsets += 1
                else:
                    cname += packet_data[current_octet*8:(current_octet+1)*8]
                    current_octet += 1
            else:
                if num_offsets > 0:
                    current_octet = offset_octet + 2
                else:
                    current_octet += 1
            packet_record_fields['rdata'] = cname

        case '0xf': #MX
            pref = '0b' + packet_data[current_octet*8:(current_octet+2)*8]
            current_octet += 2
            mx = '0b'
            num_offsets = 0
            while ('0b' + packet_data[current_octet*8:(current_octet+1)*8] != '0b00000000'):
                if ('0b' + packet_data[current_octet*8:current_octet*8+2] == '0b11'):
                    if num_offsets == 0:
                        offset_octet = current_octet
                    current_octet = int('0b' + packet_data[current_octet*8+2:(current_octet+2)*8], 2)
                    num_offsets += 1

                else:
                    mx += packet_data[current_octet*8:(current_octet+1)*8]
                    current_octet += 1
            else:
                if num_offsets > 0:
                    current_octet = offset_octet + 2
                else:
                    current_octet += 1
            packet_record_fields['rdata'] = pref, mx
            
        case _: #INVALID
            print("ERROR\t Invalid type")

    return packet_record_fields, current_octet

def read_ip(data):
     # converting the IP address to decimal    
    iterator = 0 
    result = "" #initialize ip variable 
    data = data[2:] #need to truncate the first two bits to ignore the 0b
    while iterator < len(data):
        result += str(int(data[iterator:iterator+8], 2)) + "."
        iterator += 8        
    return result[:-1]

def read_rdata(data): 
    iterator = 0
    result = "" #initialize ip variable 
    data = data[2:] #need to truncate the first two bits to ignore the 0b
    max = int(data[iterator:iterator+8], 2)
    count = 0
    iterator += 8
    while iterator < len(data):
        n = (int(data[iterator:iterator+8], 2))
        if count < max: 
            result += n.to_bytes((n.bit_length() + 7) // 8, 'big').decode() 
            count = count + 1 
        else: 
            max = int(data[iterator:iterator+8], 2)
            count = 0
            result += "."
        iterator += 8      
    return result
    

def print_record(type, rdata, seconds, auth):
    type = int(type, 2)
    type = hex(type)
    seconds = int(seconds, 2)
 
    match auth:
        case '0b0':
            auth = 'auth'
        case '0b1':
            auth = 'nonauth'
    
    match type:
        case '0x1':
            #read ip address from rdata
            ip_address = read_ip(rdata) 

            # IP <tab> [ip address] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("IP\t" + ip_address + "\t" + str(seconds) + "\t" + str(auth))

        case '0x2': 
            # Read name server from rdata
            name_server = read_rdata(rdata)
            # NS <tab> [alias] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("NS\t" + name_server + "\t" + str(seconds) + "\t" + auth)
        case '0x5':
            # Read canonical name from rdata
            alias = read_rdata(rdata)

            # CNAME <tab> [alias] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("CNAME\t" + alias + "\t" + str(seconds) + "\t" + auth)
            
        case '0xf':
            # Read mail exchange from rdata
            pref, mx_data = rdata
            dom_name = read_rdata(mx_data)
            pref = int(pref, 2)
            # MX <tab> [alias] <tab> [pref] <tab> [seconds can cache] <tab> [auth | nonauth]
            print("MX\t" + dom_name + "\t" + str(pref) + "\t" + str(seconds) + "\t" + auth)

        case _:
            print("ERROR\t Invalid type")

def read_packet(packet, id):
    # convert packet to binary
    packet = convert_bytes_to_bin(packet)
        

    packet_header_fields = parse_packet_header(packet)
    packet_question_fields, current_octet = parse_packet_questions(packet)
    # parse answers (can be multiple if ancount > 1)
    packet_answer_fields = {}
    for answer_count in range(int(packet_header_fields['ancount'], 2)):
        packet_record_fields, current_octet = parse_packet_records(packet, current_octet)
        packet_answer_fields[f'{answer_count}'] =  packet_record_fields
    
    packet_additional_fields = {}
    for additional_count in range(int(packet_header_fields['arcount'], 2)):
        packet_record_fields, current_octet = parse_packet_records(packet, current_octet)
        packet_additional_fields[f'{additional_count}'] =  packet_record_fields

    # check if id matches
    if int(packet_header_fields['id'], 2) != int(id.hex(), 16):
        print("ERROR\tID mismatch")
        return
    
    # check if response is 
    if packet_header_fields['qr'] != '0b1':
        print("ERROR\tNot a response")
        return
    
    # Check if server supports recursion
    if int(packet_header_fields['ra'], 2) != 1:
        print("ERROR\tRecursion not supported")
        return

    # Check if response is valid
    match packet_header_fields['rcode']:
        case '0b0000':
            print("") # no error
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

    # print answer section
    if (int(packet_header_fields['ancount'], 2) == 0):
        print("NOTFOUND")
        return

    print(f"***Answer Section ({int(packet_header_fields['ancount'], 2)} records)***")
    for i in range(int(packet_header_fields['ancount'], 2)):
        # Check for class mismatch (must be 1)
        if int(packet_answer_fields[f'{i}']['class'], 2) != 1:
            print("ERROR\tClass mismatch")
            return
        print_record(packet_answer_fields[f'{i}']['type'], packet_answer_fields[f'{i}']['rdata'], packet_answer_fields[f'{i}']['ttl'], packet_header_fields['aa'])
   
    if (int(packet_header_fields['arcount'], 2) != 0):
        print(f"\n***Additional Section ({int(packet_header_fields['arcount'], 2)} records)***")
        for i in range(int(packet_header_fields['arcount'], 2)):
            print_record(packet_additional_fields[f'{i}']['type'], packet_additional_fields[f'{i}']['rdata'], packet_additional_fields[f'{i}']['ttl'], packet_header_fields['aa'])
    

    
def qtype(mail_server, name_server):
    if mail_server: 
        return [0, 15]
    elif name_server:
        return [0, 2]
    else: 
        return [0, 1]

if __name__ == "__main__":
    args = collect_args()
    
    timeout = args.t
    retries = args.r
    port = args.p
    mail_server = args.mx
    name_server = args.ns
    ip_address = args.server[1:]
    domain_name = args.name
    id = random_id()

    # process type
    if mail_server:
        server_type = "MX"
    elif name_server:
        server_type = "NS"
    else:
        server_type = "A"

    # Print starting message
    print(f"DnsClient sending request for {domain_name}")
    print(f"Server: {ip_address}")
    print(f"Request type: {server_type}\n")

    #header section 
    header = create_header(id)

    #question section
    server_type = qtype(mail_server, name_server)
    question_packet = header + create_question(domain_name, server_type)
    response_packet, time = query_server(ip_address, port, timeout, retries, question_packet)

    if response_packet:
        read_packet(response_packet, id)


