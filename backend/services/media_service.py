from __future__ import annotations

import base64
from io import BytesIO
from typing import Optional

import requests
from PIL import PngImagePlugin
import piexif
import piexif.helper

from modules.shared import opts
from modules import images


class MediaService:
    """Image encode/decode helpers for API/service layers."""

    @staticmethod
    def _verify_url(url: str) -> bool:
        """Returns True if the url refers to a global resource."""
        import socket
        import ipaddress
        from urllib.parse import urlparse
        try:
            parsed_url = urlparse(url)
            domain_name = parsed_url.netloc
            host = socket.gethostbyname_ex(domain_name)
            for ip in host[2]:
                ip_addr = ipaddress.ip_address(ip)
                if not ip_addr.is_global:
                    return False
        except Exception:
            return False
        return True

    def decode_image(self, encoding: str):
        """Decode base64 or fetch from URL respecting opts.* flags."""
        if encoding.startswith("http://") or encoding.startswith("https://"):
            if not opts.api_enable_requests:
                raise ValueError("Requests not allowed")

            if opts.api_forbid_local_requests and not self._verify_url(encoding):
                raise ValueError("Request to local resource not allowed")

            headers = {'user-agent': opts.api_useragent} if opts.api_useragent else {}
            response = requests.get(encoding, timeout=30, headers=headers)
            return images.read(BytesIO(response.content))

        if encoding.startswith("data:image/"):
            encoding = encoding.split(";")[1].split(",")[1]

        return images.read(BytesIO(base64.b64decode(encoding)))

    def encode_image(self, image) -> str:
        with BytesIO() as output_bytes:
            if isinstance(image, str):
                return image
            if opts.samples_format.lower() == 'png':
                use_metadata = False
                metadata = PngImagePlugin.PngInfo()
                for key, value in image.info.items():
                    if isinstance(key, str) and isinstance(value, str):
                        metadata.add_text(key, value)
                        use_metadata = True
                image.save(output_bytes, format="PNG", pnginfo=(metadata if use_metadata else None), quality=opts.jpeg_quality)
            elif opts.samples_format.lower() in ("jpg", "jpeg", "webp"):
                if image.mode in ("RGBA", "P"):
                    image = image.convert("RGB")
                parameters = image.info.get('parameters', None)
                exif_bytes = piexif.dump({
                    "Exif": { piexif.ExifIFD.UserComment: piexif.helper.UserComment.dump(parameters or "", encoding="unicode") }
                })
                if opts.samples_format.lower() in ("jpg", "jpeg"):
                    image.save(output_bytes, format="JPEG", exif=exif_bytes, quality=opts.jpeg_quality)
                else:
                    image.save(output_bytes, format="WEBP", exif=exif_bytes, quality=opts.jpeg_quality, lossless=opts.webp_lossless)
            else:
                raise ValueError("Invalid image format")

            bytes_data = output_bytes.getvalue()
            return base64.b64encode(bytes_data).decode('utf-8')

