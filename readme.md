# Assignment 1 - Group 5
Sabrina Mansour: 260945807  
Mathieu Geoffroy: 260986559  


## Information
Python version: 3.10.2  

Run the file with the following command and arguments below: 
```
python DnsClient [-t timeout] [-r max-retries] [-p port] [-mx | nx] @server name
```

-t (5): gives how long to wait, in seconds before retransmitting an unanswered query.   
-r (3): max # of times to retransmit an unanswered query before giving up  
-p (53): UDP port number of DNS server   
-mx (IP address): send a mail server query   
-ns (IP address): send a name server query   
