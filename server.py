import socket
import threading
from pykalman import KalmanFilter
import scipy.integrate as it
import numpy as np
from KalmanFilter import kalman_filter

# Variables for holding information about connections
connections = []
total_connections = 0

isFirstIteration = True

filtered_state_means = []
filtered_state_covariances = []

actual_pos_x = [0]
actual_pos_y = [0]
actual_pos_z = [0]

def calcuate(accelerationData):

    global isFirstIteration
    global actual_pos_x
    global actual_pos_y
    global actual_pos_z
    Acc_Variance = 0.0020

    accelerationData = accelerationData[1:-1]
    accelerationData = accelerationData.split(", ")
    if(len(accelerationData) <= 3 or "[" in accelerationData[3] or "]" in accelerationData[3] or "[" in accelerationData[2] or "]" in accelerationData[2] or "[" in accelerationData[1] or "]" in accelerationData[1] or "[" in accelerationData[0] or "]" in accelerationData[0]):
        return 0
    else:
        #accelerationData[0] = float(accelerationData[0])
        #accelerationData[1] = float(accelerationData[1])
        #accelerationData[2] = float(accelerationData[2])
        #accelerationData[3] = float(accelerationData[3])
        """
        v = [0]  # or whatever you initial velocity is
        x = [0]  # or whatever you initial location is


        # time step
        dt = accelerationData[3]
        v.append(accelerationData[0] * dt)
        x.append(0.5 * accelerationData[0] * dt ** 2)
        v.append(accelerationData[1] * dt)
        x.append(0.5 * accelerationData[1] * dt ** 2)
        v.append(accelerationData[2] * dt)
        x.append(0.5 * accelerationData[2] * dt ** 2)

        v.pop(0)
        x.pop(0)


        actual_pos_x[0] = actual_pos_x[0] + x[0]
        print((actual_pos_x[0])*1000)
        #print((actual_pos_y[0] + x[1])*1000)
        #print((actual_pos_z[0] + x[2])*1000)
        #print(actual_pos_x)
        print(accelerationData)
        """

# Client class, new instance created for each connected client
# Each instance has the socket and address that is associated with items
# Along with an assigned ID and a name chosen by the client
class Client(threading.Thread):
    def __init__(self, socket, address, id, name, signal):
        threading.Thread.__init__(self)
        self.socket = socket
        self.address = address
        self.id = id
        self.name = name
        self.signal = signal

    def __str__(self):
        return str(self.id) + " " + str(self.address)

    # Attempt to get data from client
    # If unable to, assume client has disconnected and remove him from server data
    # If able to and we get data back, print it in the server and send it back to every
    # client aside from the client that has sent it
    # .decode is used to convert the byte data into a printable string
    def run(self):
        while self.signal:
            try:
                data = self.socket.recv(128)
                print(data.decode("utf-8"))
            except:
                print("Client " + str(self.address) + " has disconnected")
                self.signal = False
                connections.remove(self)
                break
            if data != "":
                calcuate(data.decode("utf-8"))
                for client in connections:
                    if client.id != self.id:
                        client.socket.sendall(data)


# Wait for new connections
def newConnections(socket):
    while True:
        sock, address = socket.accept()
        global total_connections
        connections.append(Client(sock, address, total_connections, "Name", True))
        connections[len(connections) - 1].start()
        print("New connection at ID " + str(connections[len(connections) - 1]))
        total_connections += 1


def main():
    # Get host and port
    host = input("Host: ")
    port = int(input("Port: "))

    # Create new server socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.bind((host, port))
    sock.listen(5)

    # Create new thread to wait for connections
    newConnectionsThread = threading.Thread(target=newConnections, args=(sock,))
    newConnectionsThread.start()


main()