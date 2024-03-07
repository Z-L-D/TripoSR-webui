import os
from typing import Type

import onnxruntime as ort

from .sessions import sessions_class
from .sessions.base import BaseSession
from .sessions.dis_anime import DisSession
from .sessions.dis_general_use import DisSessionGeneralUse
from .sessions.sam import SamSession
from .sessions.silueta import SiluetaSession
from .sessions.u2net_cloth_seg import Unet2ClothSession
from .sessions.u2net_human_seg import U2netHumanSegSession
from .sessions.u2net import U2netSession
from .sessions.u2netp import U2netpSession


def new_session(
    model_name: str = "u2net", providers=None, *args, **kwargs
) -> BaseSession:
    """
    Create a new session object based on the specified model name.

    This function searches for the session class based on the model name in the 'sessions_class' list.
    It then creates an instance of the session class with the provided arguments.
    The 'sess_opts' object is created using the 'ort.SessionOptions()' constructor.
    If the 'OMP_NUM_THREADS' environment variable is set, the 'inter_op_num_threads' option of 'sess_opts' is set to its value.

    Parameters:
        model_name (str): The name of the model.
        providers: The providers for the session.
        *args: Additional positional arguments.
        **kwargs: Additional keyword arguments.

    Returns:
        BaseSession: The created session object.
    """
    if model_name == "dis_anime":
        session_class: Type[BaseSession] = DisSession
    if model_name == "dis_general_use":
        session_class: Type[BaseSession] = DisSessionGeneralUse
    if model_name == "sam":
        session_class: Type[BaseSession] = SamSession
    if model_name == "silueta":
        session_class: Type[BaseSession] = SiluetaSession
    if model_name == "u2net_cloth_seg":
        session_class: Type[BaseSession] = Unet2ClothSession
    if model_name == "u2net_human_seg":
        session_class: Type[BaseSession] = U2netHumanSegSession
    if model_name == "u2net":
        session_class: Type[BaseSession] = U2netSession
    if model_name == "u2netp":
        session_class: Type[BaseSession] = U2netpSession

    # session_class: Type[BaseSession] = U2netSession

    for sc in sessions_class:
        if sc.name() == model_name:
            session_class = sc
            break

    sess_opts = ort.SessionOptions()

    if "OMP_NUM_THREADS" in os.environ:
        sess_opts.inter_op_num_threads = int(os.environ["OMP_NUM_THREADS"])
        sess_opts.intra_op_num_threads = int(os.environ["OMP_NUM_THREADS"])

    return session_class(model_name, sess_opts, providers, *args, **kwargs)
