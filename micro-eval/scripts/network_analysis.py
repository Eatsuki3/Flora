import re
import sys

# Define a regular expression pattern to extract port and length information
pattern = re.compile(
    r'(\d+:\d+:\d+\.\d+) IP [\d.]+\.(\d+) > [\d.]+\.(\d+): .*? length (\d+)')

# Specify the path to your tcpdump output file
if len(sys.argv) != 2:
    print("Usage: python3 network_analysis.py <tcpdump_output>")
    sys.exit(1)
file_path = sys.argv[1]

traffic_by_port = {'mpc': 0, 'non_mpc': 0}

matched_line = 0

# Party setup/listen ports (ip_1/ip_2/ip_3 in the configs)
non_mpc_ports = {5555, 5556, 5557}

# 2PC MPC ports: start_port .. start_port+512 (12345..12857)
min_mpc_port = 12345
max_mpc_port = 12345 + 512

# Dealer ports: dealer_start_port .. dealer_start_port+DEALER_PORT_GAP+filter_streams
# (30000..35100); classify the whole 30000-35200 range as MPC.
min_dealer_port = 30000
max_dealer_port = 35200

total_length = 0

# Open and read the tcpdump output file
with open(file_path, 'r') as file:
    for line in file:
        match = pattern.search(line)
        if match:
            source_port = int(match.group(2))
            destination_port = int(match.group(3))
            length = int(match.group(4))

            non_mpc = (source_port in non_mpc_ports or
                       destination_port in non_mpc_ports)

            mpc = ((min_mpc_port <= source_port <= max_mpc_port or
                    min_mpc_port <= destination_port <= max_mpc_port) or
                   (min_dealer_port <= source_port <= max_dealer_port or
                    min_dealer_port <= destination_port <= max_dealer_port))

            if non_mpc:
                traffic_by_port['non_mpc'] += length
            if mpc:
                traffic_by_port['mpc'] += length

            matched_line += 1
            total_length += length

            if not (mpc or non_mpc):
                print("unclassified line:", line.strip())
        else:
            print('unmatched line', line.strip())

print(traffic_by_port)
print('total length', total_length)
print("matched lines:", matched_line)
