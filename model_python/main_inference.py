import time
from data_preparation import getPicList
from layer_computation import (compute_conv_layer, compute_bn_layer, compute_relu_layer, compute_maxpool_layer,
compute_bottleneck, compute_avgpool_layer, compute_fc_layer)
from data_preparation import preprocess


pic_to_predice = getPicList()

for filename in pic_to_predice:
    print("begin predice with " + filename)
    
    start_time = time.time()
    out = preprocess(filename)
    print(f"Time taken for preprocess: {time.time() - start_time:.6f} seconds")
    
    start_time = time.time()
    out = compute_conv_layer(out, "conv1")
    print(f"Time taken for compute_conv_layer: {time.time() - start_time:.6f} seconds")

    start_time = time.time()
    out = compute_bn_layer(out, "bn1")
    print(f"Time taken for compute_bn_layer: {time.time() - start_time:.6f} seconds")
    
    out = compute_relu_layer(out)
    
    out = compute_maxpool_layer(out)

    # layer1 
    out = compute_bottleneck(out, "layer1_bottleneck0", down_sample=True)
    out = compute_bottleneck(out, "layer1_bottleneck1", down_sample=False)
    out = compute_bottleneck(out, "layer1_bottleneck2", down_sample=False)

    # layer2
    out = compute_bottleneck(out, "layer2_bottleneck0", down_sample=True)
    out = compute_bottleneck(out, "layer2_bottleneck1", down_sample=False)
    out = compute_bottleneck(out, "layer2_bottleneck2", down_sample=False)
    out = compute_bottleneck(out, "layer2_bottleneck3", down_sample=False)

    # layer3
    out = compute_bottleneck(out, "layer3_bottleneck0", down_sample=True)
    out = compute_bottleneck(out, "layer3_bottleneck1", down_sample=False)
    out = compute_bottleneck(out, "layer3_bottleneck2", down_sample=False)
    out = compute_bottleneck(out, "layer3_bottleneck3", down_sample=False)
    out = compute_bottleneck(out, "layer3_bottleneck4", down_sample=False)
    out = compute_bottleneck(out, "layer3_bottleneck5", down_sample=False)

    # layer4
    out = compute_bottleneck(out, "layer4_bottleneck0", down_sample=True)
    out = compute_bottleneck(out, "layer4_bottleneck1", down_sample=False)
    out = compute_bottleneck(out, "layer4_bottleneck2", down_sample=False)

    # avg pool
    start_time = time.time()
    out = compute_avgpool_layer(out)
    print(f"Time taken for compute_avgpool_layer: {time.time() - start_time:.6f} seconds")

    # Linear
    start_time = time.time()
    out = compute_fc_layer(out, "fc")
    print(f"Time taken for compute_fc_layer: {time.time() - start_time:.6f} seconds")

    # find inference result
    start_time = time.time()
    out_res = list(out)
    max_value = max(out_res)
    index = out_res.index(max_value)
    print(f"Time taken for finding inference result: {time.time() - start_time:.6f} seconds")

    print("\npredict picture: " + filename)
    print("\npredict picture: " + filename)
    print("      max_value: " + str(max_value))
    print("          index: " + str(index))

    # Read the categories
    start_time = time.time()
    with open("imagenet_classes.txt", "r") as f:
        categories = [s.strip() for s in f.readlines()]
    print(f"Time taken for reading categories: {time.time() - start_time:.6f} seconds")

    print("         result: " + categories[index])
