import socket
import Adafruit_DHT
import json 

sensor = Adafruit_DHT.DHT11
pin = 4

humidity, temperature = Adafruit_DHT.read_retry(sensor, pin)

host = ''
port = 8081

server_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
server_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)

server_socket.bind((host, port))
server_socket.listen()

while True:
    client_socket, client_addr = server_socket.accept()

    print("[+] Client {} is connected".format(client_addr))

    if humidity is not None or temperature is not None:
        client_socket.send(json.dumps({"Temp": f"{temperature:0.1f}",
                                    "Humid": f"{humidity:0.1f}"}).encode() )
    else:
        client_socket.send("Error")
    client_socket.close()