# sparse moe with different structure
two router:
one for adapter type (lora QKVO FFN_UP FFN_DOWN, prompt, parallel adapter)
one for experts of same type(topk moe, adamole) 利用adapter router adapted_weight（weight-thresh） 作为adamole的参数调控expert数量
