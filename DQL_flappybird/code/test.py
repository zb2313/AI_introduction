import argparse
import torch

from src.deep_q_network import DeepQNetwork
from src.flappy_bird import FlappyBird
from src.utils import pre_processing


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Flappy Bird""")
    parser.add_argument("--image_size", type=int, default=84, help="The common width and height for all images")
    parser.add_argument("--saved_path", type=str, default="trained_models")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/flappy_bird".format(opt.saved_path))#加载以及训练好的模型 无gpu
    else:
        model = torch.load("{}/flappy_bird".format(opt.saved_path), map_location=lambda storage, loc: storage)#有gpu
    model.eval()
    game_state = FlappyBird()
    image, reward, terminal = game_state.next_frame(0)#加载图片 奖励值 终端
    image = pre_processing(image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size, opt.image_size)#修改图片格式
    image = torch.from_numpy(image)
    if torch.cuda.is_available():
        model.cuda()
        image = image.cuda()
    state = torch.cat(tuple(image for _ in range(4)))[None, :, :, :]#状态表

    while True:
        prediction = model(state)[0]#根据状态输出预测
        print('predict',prediction)
        action = torch.argmax(prediction)
        print('action',action)
        next_image, reward, terminal = game_state.next_frame(action)#下一个游戏帧
        next_image = pre_processing(next_image[:game_state.screen_width, :int(game_state.base_y)], opt.image_size,
                                    opt.image_size)
        next_image = torch.from_numpy(next_image)
        if torch.cuda.is_available():
            next_image = next_image.cuda()
        next_state = torch.cat((state[0, 1:, :, :], next_image))[None, :, :, :]#下一个游戏状态

        state = next_state

if __name__ == "__main__":
    opt = get_args()#设置初始参数
    test(opt)
