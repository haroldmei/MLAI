import torch

from highway import Highway
from cnn import CNN


## sanity check code
def testHighway():
    size = 5

    hw = Highway(size)
    t = torch.randn((5,5))

    xproj = hw.ReLU_W_proj(t)
    assert t.size() == xproj.size()
    print("Sanity Check xproj shape for highway passed")

    xgate = hw.Sigmoid_W_gate(t)
    assert t.size() == xgate.size()
    print("Sanity Check xgate shape for highway passed")

    x_highway = xproj * xgate +(1 - xgate) * t
    assert t.size() == x_highway.size()
    print("Sanity Check x_highway shape for highway passed")

    out = hw(t)    
    assert t.size() == out.size()
    print("Sanity Check Input/Output shape for highway passed")


## sanity check code
def testCNN():
    in_channel= 5
    out_channel = 4
    k = 2

    cnn = CNN(in_channel, k, out_channel)
    input = torch.randn((1,5,1))
    
    t = cnn.Conv1d(input)
    assert torch.Size([1, out_channel,k]) == t.size()
    print("Sanity Check conv1d shape passed for CNN")

    t = torch.max(t, dim=2)[0]
    assert torch.Size([1, out_channel]) == t.size()
    print("Sanity Check maxpool shape passed for CNN")

    out = cnn(input)
    assert torch.Size([1, out_channel]) == out.size()
    print("Sanity Check Input/Output shape passed for CNN")

if __name__ == '__main__':
    
    testHighway()
    testCNN()