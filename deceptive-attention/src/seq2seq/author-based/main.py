import argparse

from train import train


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument('--task', dest='task', default='en-de',
                        choices=('copy', 'reverse-copy', 'binary-flip', 'en-hi', 'en-de'),
                        help='select the task you want to run on')

    parser.add_argument('--debug', dest='debug', action='store_true')
    parser.add_argument('--loss-coef', dest='loss_coeff', type=float, default=0.0)
    parser.add_argument('--epochs', dest='epochs', type=int, default=5)
    parser.add_argument('--seed', dest='seed', type=int, default=1234)

    parser.add_argument('--attention', dest='attention', type=str, default='dot-product')

    parser.add_argument('--batch-size', dest='batch_size', type=int, default=128)
    parser.add_argument('--num-train', dest='num_train', type=int, default=1000000)
    parser.add_argument('--decode-with-no-attn', dest='no_attn_inference', action='store_true')

    parser.add_argument('--tensorboard_log', dest='tensorboard_log', action='store_true')

    params = vars(parser.parse_args())

    train(params['task'],
          params['loss_coeff'],
          params['seed'],
          params['attention'],  # can have values 'dot-product', 'uniform', or 'no-attention'
          params['epochs'],
          params['batch_size'],
          params['no_attn_inference'],
          params['tensorboard_log'],
          params['debug'],
          params['num_train'])


if __name__ == "__main__":
    main()
