# -*- coding:utf-8 -*-
from .base_options import BaseOptions


class TestOptions(BaseOptions):
    """This class includes test options.

    It also includes shared options defined in BaseOptions.
    """

    def initialize(self, parser):
        parser = BaseOptions.initialize(self, parser)  # define shared options
        parser.add_argument('--ntest', type=int, default=float("inf"), help='# of test examples.')
        parser.add_argument('--test_img_path', type=str, default='  ', help='path of single test image.')
        parser.add_argument('--test_upscale', type=int, default=1, help='upscale single test image.')
        parser.add_argument('--results_dir', type=str, default='./results/', help='saves results here.')
        parser.add_argument('--save_as_dir', type=str, default='/home/wang107552002794/Ours/result_helen/SPARNet_test/', help='save results in different dir.')
        parser.add_argument('--phase', type=str, default='test', help='train, val, test, etc')
        parser.add_argument('--pretrain_model_path', type=str, default='/home/wang107552002794/Ours/pretrain_models/Ours.pth', help='load pretrain model path if specified')
        parser.add_argument('--eval', action='store_true', help='use eval mode during test time.')
        parser.add_argument('--num_test', type=int, default=50, help='how many test images to run')
        parser.set_defaults(model='Ours')

        self.isTrain = False
        return parser
