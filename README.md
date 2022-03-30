# Adversarial Robusteness in Distributed Systems
## Federated Learning with PGD Adversarial Training


Federated Learning is a privacy preserving machine learning setting where models are trained across decentralized client devices and the model weights are sent to the server for aggregation. This ensures high performance on the client device without sharing the data with the server. 

It has been shown that deep neural netwroks can be easliy fooled and are prone to inference-time Whitebox Adversarial Attacks. Adversarial Trianing against the PGD (Projected Gradient Descent) adversary has been proved to be highly effective against these attacks.   

This repo contains a Pytoch implementation to replicate a simple FL setting along with an option to do adversarial training across the clients.


### How to run the Code 

- List of training parameters
    ```
    python main.py -h
    ```

- Runing the code with default parameters.

    ``` 
    python main.py
    ```
