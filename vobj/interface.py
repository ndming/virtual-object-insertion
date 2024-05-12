import cv2, random
import numpy as np
from PIL.Image import Image


def prompt_image_point(
    image: Image | np.ndarray, 
    n_points: int | tuple[int, int], 
    window_name=""
) -> np.ndarray:
    """
    Prompts a window with the `image` and listens for `n_points` clicks. If 
    `n_points` is given as a tuple (`min`, `max`), the function waits for point 
    click event until an ENTER or an interrupt event is signaled. The number of 
    received image points will be validated against the provided `n_points`.

    Parameters:
    ---
    - `image`: the 2D image array, guaranteed to be intact throughout.
    - `n_points`: an integer or a tuple specifying the required number of points.
    - `window_name`: the string to be shown on the window title bar.

    Returns:
    ---
    An `n-by-2` array containing the selected `n_points` pixel coordinates.
    """

    # Flip the image since opencv displays images in BGR
    img = np.copy(np.flip(np.array(image), axis=2))

    if not window_name:
        window_name = f"window-{hex(random.randint(0, 2**32))}"
    
    points = []
    callback_param = (img, points, window_name)
    cv2.imshow(window_name, img)
    cv2.setMouseCallback(window_name, _select_point_callback, callback_param)
    
    if isinstance(n_points, int):
        while cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) > 0:
            cv2.waitKey(1)
            if len(points) >= n_points:
                cv2.destroyAllWindows()

        if len(points) < n_points:
            raise ValueError(len(points))
    else:
        min, max = n_points
        cv2.waitKey()
        cv2.destroyAllWindows()
        if len(points) > max or len(points) < min:
            raise ValueError(len(points))

    return np.asarray(points)


def _select_point_callback(
    event, x, y, _, param: tuple[np.ndarray, list[tuple], str]):
    if event == cv2.EVENT_LBUTTONDOWN:
        img, points, window_name = param
        points.append((x, y))
        cv2.circle(img, (x, y), 4, (255, 0, 0), -1)
        cv2.imshow(window_name, img)