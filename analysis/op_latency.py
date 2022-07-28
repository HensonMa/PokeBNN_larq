graph_info_file = "PokeBNN_1x_compute_graph.txt"
latency_info_file = "PokeBNN_1x_latency.txt"



class LatencyTable(object):
    def __init__(self, latency_info_file):
        self.latency_info_file = latency_info_file
        self.latency_table = {}
        self.parse_latency_info_file()

    def parse_latency_info_file(self):
        lines = open(self.latency_info_file).readlines()
        op_profile_lines = list()
        # select the lines that contain the operator-wise profiling info
        interested = False
        for line in lines:
            if "Top by Computation Time" in line:
                interested = False
            if interested and line != "\n":
                op_profile_lines.append(line)
            if "Operator-wise Profiling" in line:
                interested = True
        op_profile_lines = op_profile_lines[2:] # lose the header
        # get the [avg ms] and [Name] from the lines
        for line in op_profile_lines:
            name = line.split('[')[1].split(']')[0]
            names = name.split(',')
            line = line.strip()
            line = line.split()
            avg_ms = float(line[3])
            for n in names:
                n = n.strip()
                self.latency_table[n] = avg_ms
                # some ops are combined into one,
                # so we only add avg_ms to the first one
                # the rest will be set to zero
                avg_ms = 0

    def get_latency(self, op_name):
        if op_name in self.latency_table:
            return self.latency_table[op_name]
        else:
            raise Exception("op_name: {} not found in latency table".format(op_name))

    def print_latency_table(self):
        for k, v in self.latency_table.items():
            print("{}: {} ms".format(k, v))    


class ComputeGraph(object):
    def __init__(self, graph_info_file):
        self.graph_info_file = graph_info_file
        self.graph_table = {}
        self.parse_graph_info_file()

    def parse_graph_info_file(self):
        pass

    def get_op_latency(self, op_name):
        pass


class OpLatencyTable(object):
    def __init__(self):
        self.latency_table = {}

    def add_op_latency(self, op_name, latency):
        self.latency_table[op_name] = latency



latency_table = LatencyTable(latency_info_file)