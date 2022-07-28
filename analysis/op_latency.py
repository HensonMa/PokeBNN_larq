graph_info_file = "PokeBNN_1x_compute_graph.txt"
latency_info_file = "PokeBNN_1x_latency.txt"



class LatencyTable(object):
    def __init__(self, latency_info_file):
        self.latency_info_file = latency_info_file
        self.latency_table = {}
        self.parse_latency_info_file()

    def parse_latency_info_file(self):
        pass


class ComputeGraph(object):
    def __init__(self, graph_info_file):
        self.graph_info_file = graph_info_file
        self.graph_table = {}
        self.parse_graph_info_file()

    def parse_graph_info_file(self):
        pass

    def get_op_latency(self, op_name):
        pass