import base64
import unittest
from unittest.mock import patch
from app import image_recognition, model

PNG = b'\x89PNG\r\n\x1a\nexample'

class ImageRecognitionTests(unittest.TestCase):
    def test_http_uses_configured_backend_and_returns_reviewable_text(self):
        config = {'backend': 'http'}
        with patch.object(image_recognition.settings, 'resolve', return_value=config), patch.object(model, '_generate_http', return_value={'raw': 'x^2'}) as call:
            result = image_recognition.recognize('a.png', base64.b64encode(PNG).decode())
        self.assertEqual(result['text'], 'x^2')
        self.assertIn('data:image/png;base64,', call.call_args.kwargs['image_data_url'])
        self.assertIn('核对', result['warning'])

    def test_rejects_invalid_images_before_model_call(self):
        with patch.object(model, '_generate_http') as call:
            with self.assertRaises(ValueError):
                image_recognition.recognize('a.png', base64.b64encode(b'not an image').decode())
            call.assert_not_called()

    def test_unsupported_backend_does_not_silently_fallback(self):
        with patch.object(image_recognition.settings, 'resolve', return_value={'backend': 'dsh-cli'}):
            with self.assertRaises(model.ModelError):
                image_recognition.recognize('a.png', base64.b64encode(PNG).decode())
