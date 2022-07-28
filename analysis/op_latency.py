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

    def get_latency(self, tensor_name):
        """
        key: tensor_name
        value: latency
        """
        if tensor_name in self.latency_table:
            return self.latency_table[tensor_name]
        else:
            raise Exception("op_name: {} not found in latency table".format(tensor_name))

    def get_tensor_list(self):
        return self.latency_table.keys()

    def print_latency_table(self):
        for k, v in self.latency_table.items():
            print("{}: {} ms".format(k, v))    

class Op(object):
    def __init__(self, name, op_type):
        self.name = name
        self.type = op_type


class ComputeGraph(object):
    """
    Assumption: every op produces a single tensor
    """
    def __init__(self, graph_info_file):
        self.graph_info_file = graph_info_file
        self.tensoridx_to_op = {}
        self.tensorname_to_idx = {}
        self.parse_graph_info_file()


    def parse_graph_info_file(self):
        lines = open(self.graph_info_file).readlines()
        op_lines = list()
        tensor_lines = list()
        # select the lines that have op info
        for line in lines:
            if "Op#" in line:
                op_lines.append(line)
        # select the lines that have tensor info
        for line in lines:
            if "T#" in line and "shape" in line and "type" in line: 
                tensor_lines.append(line)

        # build table: tensor_idx -> op
        for line in op_lines:
            # e.g.   Op#1247 RESHAPE(T#1473, T#78) -> [T#1474]
            result_tensor = line.split("->")[1].strip()
            result_tensor = result_tensor.split("[")[1].split("]")[0]
            if "," in result_tensor:
                raise Exception("multiple tensors are produced by one op, not supported")
            tensor_idx = result_tensor.strip()
            op_type = line.split("(")[0].split(" ")[-1]
            op_idx = line.split(" ")[0]
            op = Op(op_idx, op_type)
            self.tensoridx_to_op[tensor_idx] = op

        
        # build table: tensor_name -> tensor_idx
        for line in tensor_lines:
            # e.g. T#46(dp_re_lu/beta) shape:[1, 1, 1, 32], type:FLOAT32 RO 128 bytes
            tensor_name = line.split("(")[1].split(")")[0]
            tensor_idx = line.split("(")[0]
            tensor_idx = tensor_idx.strip()
            self.tensorname_to_idx[tensor_name] = tensor_idx


    def get_op_type(self, tensor_name):
        """
        key: tensor name
        value: op type
        """
        if tensor_name in self.tensorname_to_idx:
            tensor_idx = self.tensorname_to_idx[tensor_name]
            if tensor_idx in self.tensoridx_to_op:
                return self.tensoridx_to_op[tensor_idx].type
            else:
                raise Exception("tensor_name: `{}` not found in tensoridx_to_op".format(tensor_idx))
        else:
            raise Exception("tensor_name: `{}` not found in tensorname_to_idx".format(tensor_name))

    def print_table(self):
        # note that #tensor is larger than #op
        # because some are constant tensors
        print("tensorname_to_idx:")
        for k, v in self.tensorname_to_idx.items():
            print("{}: {}".format(k, v))
        print("tensoridx_to_op:")
        for k, v in self.tensoridx_to_op.items():
            print("{}: {}".format(k, v.type))


class OpLatencyTable(object):
    def __init__(self):
        self.latency_table = {}

    def add_op_latency(self, op_name, latency):
        if op_name in self.latency_table:
            self.latency_table[op_name] += latency
        else:
            self.latency_table[op_name] = latency

    def report(self):
        print("Op latency table:")
        print("----------------------------------------------------")
        for k, v in self.latency_table.items():
            print("{}: {} ms".format(k, v))
        print("----------------------------------------------------")
        print("total latency: {} ms".format(sum(self.latency_table.values())))


# tensor name -> latency
latency_table = LatencyTable(latency_info_file)
# tensor name -> op type
compute_graph = ComputeGraph(graph_info_file)
# op type -> latency
op_latency_table = OpLatencyTable()
for tensor in latency_table.get_tensor_list():
    # get the op type
    op_type = compute_graph.get_op_type(tensor)
    # get the latency
    latency = latency_table.get_latency(tensor)
    # print("{}: {} ms".format(op_type, latency))
    # add the latency to the op latency table
    op_latency_table.add_op_latency(op_type, latency)

op_latency_table.report()