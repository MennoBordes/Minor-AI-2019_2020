import socket
MCAST_GRP = "145.137.9.149"
MCAST_PORT = 49000

sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM, socket.IPPROTO_UDP)
sock.setsockopt(socket.IPPROTO_IP, socket.IP_MULTICAST_TTL, 2)
sock.sendto(b"test", (MCAST_GRP, MCAST_PORT))
