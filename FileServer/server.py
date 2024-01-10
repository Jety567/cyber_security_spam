import http.server
import socketserver
import os

# Set the directory you want to serve files from
directory_to_serve = '/Users/patrickweber/PycharmProjects/spam_classification/FileServer/Files'

# Set the port for the server
port = 8000

# Change to the specified directory
os.chdir(directory_to_serve)

# Create a simple HTTP server
handler = http.server.SimpleHTTPRequestHandler

# Set up the server on the specified port
httpd = socketserver.TCPServer(("", port), handler)

# Print a message indicating that the server is running
print(f"Serving files from {directory_to_serve} on port {port}")

# Start the server
httpd.serve_forever()
