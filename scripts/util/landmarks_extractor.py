import inspect

from skimage import io
import face_alignment


def _no_compile_kwargs():
    """Disable face_alignment's torch.compile, when that build supports it.

    The master branch of face-alignment compiles its networks by default
    (`compile=True`). That needs Triton, which the Windows PyTorch wheels do
    not ship, and its warm-up failure handler logs "using eager mode" without
    restoring the uncompiled network - so every later forward raises
    BackendCompilerFailed. The released 1.4.1 has no such argument.
    """
    params = inspect.signature(face_alignment.FaceAlignment.__init__).parameters
    return {"compile": False} if "compile" in params else {}


class LandmarksExtractor:
    def __init__(self, device="cuda", landmarks_type="2D", flip=False):
        self.fa = face_alignment.FaceAlignment(
            face_alignment.LandmarksType.TWO_D
            if landmarks_type == "2D"
            else face_alignment.LandmarksType.THREE_D,
            flip_input=flip,
            device=device,
            face_detector="sfd",
            **_no_compile_kwargs(),
        )

        self.landmarks = []

    def cuda(self):
        return self

    def extract_landmarks(self, image):
        # image: either a path to an image or a numpy array (H, W, C) or tensor batch  (B, C, H, W)
        if isinstance(image, str):
            image = io.imread(image)
        if len(image.shape) == 3:
            preds = self.fa.get_landmarks(image)
        else:
            preds = self.fa.get_landmarks_from_batch(image)

        return preds
