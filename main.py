import argparse
import server
import user
import numpy as np
import json


def federated_train(dataset, model, learning_rate, batch_size=128, users_count=11, epochs=150,
                                                         resume=0, checkpoint=100, adversarial_training=0, momentum=0.9):
    print(locals())
    users = []
    for user_id in range(users_count):
        users.append(user.User(user_id, batch_size, users_count, momentum, dataset, model, adversarial_training))

    the_server = server.Server(users, batch_size, learning_rate, 0, momentum, data_set=dataset, model=model)
    test_size = len(the_server.test_loader.dataset)

    print("\nStarting Training...")

    TEST_STEP = 5

    clean_acc = []
    adv_acc = []
    for epoch in range(resume, epochs):
        if resume:
            the_server.load_model(resume)
            resume = 0
            print("Checkpoint loaded from epoch: ", epoch)
        
        if epoch == checkpoint:
            the_server.save_model(epoch)

        the_server.train_client(epoch)
        the_server.collect_gradients()
        the_server.fedAvg()
        
        if epoch % TEST_STEP == 0:
            test_loss, correct = the_server.test()
            accuracy = 100. * float(correct) / test_size
            print('Epoch: [{:3d}] Average loss: {:.4f}, Accuracy: ({:.2f}%)'.format(epoch, test_loss, accuracy))
            clean_acc.append((accuracy, epoch))

        if (epoch % TEST_STEP==0):
            adv_test_loss, adv_correct = the_server.adv_test(0.1)
            adv_accuracy = 100. * float(adv_correct) / test_size
            print('Epoch: [{:3d}] Average loss: {:.4f}, Adv Acc: ({:.2f}%)'.format(epoch, adv_test_loss, adv_accuracy))        
            adv_acc.append((adv_accuracy, epoch))
    
    res = {}
    res['clean_acc'] = clean_acc
    res['adv_acc'] = adv_acc
    
    with open('result_vgg.json', 'w') as f:
        json.dump(res, f)
    
    the_server.save_model(epochs)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Adversarial Robusteness in Federated Learning Learning')
    parser.add_argument('-s', '--dataset', default='CIFAR10', choices=['MNIST', 'CIFAR10', 'CIFAR100'], help='image dataset')
    parser.add_argument('-m', '--model', default='VGG16', choices=['VGG16', 'AlexNet'], help='model architecture')
    parser.add_argument('-n', '--users-count', default=11, type=int,help='number of clients')
    parser.add_argument('-b', '--batch_size', default=128, type=int,help='batch_size')
    parser.add_argument('-e', '--epochs', default=200, type=int, help='number of epochs')
    parser.add_argument('-l', '--learning_rate', default=0.01, type=float,help='initial learning rate')
    parser.add_argument('-r', '--resume', default=0, type=int, help='resume training from some checkpoint')
    parser.add_argument('-ckpt', '--checkpoint', default=100, type=int, help='create a checkpoint at this epoch')
    parser.add_argument('-adv', '--adversarial_training', default=0, type=bool, help='whether to do adversarial training on client side')
    args = parser.parse_args()
    
    federated_train(args.dataset, args.model, args.learning_rate, args.batch_size, args.users_count, args.epochs, 
                            resume = args.resume, checkpoint=args.checkpoint, adversarial_training=args.adversarial_training)
