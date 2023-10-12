from .gradients import get_gradients
from data import split_data_and_move_to_device

def woodfisher(network, data_loader, criterion, v, B=None):
    device = next(network.parameters()).device
    network.eval()
    k_new = v
    N = len(data_loader) if B == None else B
    for idx, data in enumerate(data_loader):
        x, y = split_data_and_move_to_device(data, device)
        #x, y = x.to(device), y.to(device)
        assert x.size(0) == 1
        fx = network(x)
        loss = criterion(fx, y)
        gradients = get_gradients(network, loss)
        if idx == 0:
            o_new = gradients
        else:
            o_old = o_new
            k_old = k_new
            tmp = o_old @ gradients
            o_new = o_old - (tmp/(N+tmp))*o_old
            k_new = k_old - (tmp/(N+tmp))*k_old
        if idx == N:
            break
    return k_new