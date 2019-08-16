

class ModeKeys(object):
    TRAIN = 'train'
    EVAL = 'eval'
    PREDICT = 'infer'
    PREDICT_RL = "predict_rl"
    PREDICT_ONE = 'infer_one'
    PREDICT_ONE_ENCODER = 'infer_one_encoder'
    PREDICT_ONE_DECODER = 'infer_one_decoder'
    GENERATOR = "generator"
    DISCRIMINATOR = "discriminator"

    @staticmethod
    def is_predict_one(m):
        return m in [ModeKeys.PREDICT_ONE, ModeKeys.PREDICT_ONE_DECODER, ModeKeys.PREDICT_ONE_ENCODER]

    @staticmethod
    def is_predict(m):
        return ModeKeys.is_predict_one(m) or m in [ModeKeys.PREDICT]


