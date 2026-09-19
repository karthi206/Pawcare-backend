from PIL import Image

ALLOWED_IMAGE_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
ALLOWED_IMAGE_MIME_TYPES = {'image/png', 'image/jpeg', 'image/webp'}
MIN_IMAGE_DIMENSION = 64      # px
MAX_IMAGE_DIMENSION = 6000    # px


def allowed_extension(filename):
    return '.' in filename and \
        filename.rsplit('.', 1)[1].lower() in ALLOWED_IMAGE_EXTENSIONS


def validate_image_file(file_storage):
    if not file_storage or not file_storage.filename:
        return False, "No file provided"

    if not allowed_extension(file_storage.filename):
        return False, "Unsupported file type. Allowed: PNG, JPG, JPEG, WEBP"

    if file_storage.mimetype and file_storage.mimetype not in ALLOWED_IMAGE_MIME_TYPES:
        return False, "Unsupported file type"

    try:
        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        img.verify()

        file_storage.stream.seek(0)
        img = Image.open(file_storage.stream)
        width, height = img.size
        if width < MIN_IMAGE_DIMENSION or height < MIN_IMAGE_DIMENSION:
            return False, f"Image too small (minimum {MIN_IMAGE_DIMENSION}x{MIN_IMAGE_DIMENSION}px)"
        if width > MAX_IMAGE_DIMENSION or height > MAX_IMAGE_DIMENSION:
            return False, f"Image too large (maximum {MAX_IMAGE_DIMENSION}x{MAX_IMAGE_DIMENSION}px)"

        file_storage.stream.seek(0)
    except Exception:
        return False, "File is not a valid image"

    return True, None
