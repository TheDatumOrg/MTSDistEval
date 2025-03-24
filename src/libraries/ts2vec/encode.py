import time
import datetime
from src.libraries.ts2vec.ts2vec import TS2Vec
from src.libraries.ts2vec.utils import init_dl_program, data_dropout
from src.libraries.ts2vec.parameters import Parameters
from src.libraries.utils import channel_normalize
import gc
from torch import cuda

def save_checkpoint_callback(
    run_dir,
    save_every=1,
    unit='epoch',
):
    assert unit in ('epoch', 'iter')
    def callback(model, loss):
        n = model.n_epochs if unit == 'epoch' else model.n_iters
        if n % save_every == 0:
            model.save(f'{run_dir}/model_{n}.pkl')
    return callback

def encode(X_train, X_test, **kwargs):
    params = Parameters(**kwargs)
    device = init_dl_program(params.gpu, seed=params.seed, max_threads=params.max_threads)

    # Swap channel and time axes
    X_train = X_train.swapaxes(1, 2)
    X_test = X_test.swapaxes(1, 2)
    
    if params.irregular > 0:
        X_train = data_dropout(X_train, params.irregular)
        X_test = data_dropout(X_test, params.irregular)
    
    config = dict(
        batch_size=params.batch_size,
        lr=params.lr,
        output_dims=params.repr_dims,
        max_train_length=params.max_train_length
    )

    # DEBUG
    # X_train, _, X_test, _ = datautils.load_UEA("BasicMotions")

    # if params.save_every is not None:
    #     unit = 'epoch' if params.epochs is not None else 'iter'
    #     config[f'after_{unit}_callback'] = save_checkpoint_callback(params.save_every, unit)

    # run_dir = 'training/' + params.dataset + '__' + name_with_datetime(params.run_name)
    # os.makedirs(run_dir, exist_ok=True)

    # Normalize data
    X_train, X_test = channel_normalize(X_train, X_test, channel_axis=-1)

    # Flush cuda memory first
    gc.collect()
    cuda.empty_cache()
    
    t = time.time()
    
    model = TS2Vec(
        input_dims=X_train.shape[-1],
        device=device,
        **config
    )
    loss_log = model.fit(
        X_train,
        n_epochs=params.epochs,
        n_iters=params.iters,
        verbose=True
    )

    t = time.time() - t
    print(f"\nTraining time: {datetime.timedelta(seconds=t)}\n")

    print("Encoding train data...")
    train_repr = model.encode(X_train, encoding_window='full_series')

    print("Encoding test data...")
    test_repr = model.encode(X_test, encoding_window='full_series')

    return train_repr, test_repr