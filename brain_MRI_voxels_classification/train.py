import numpy as np

if __name__ == '__main__':
    dataset = create_dataset()
    model = create_model()
    for epoch in NUM_OF_EPOCHS:
        input = random_batch(dataset)
        model.set_input(input)
        model.forward()
        model.calc_loss()
        model.calc_gradient()
        model.update_weights()
    save_network()
