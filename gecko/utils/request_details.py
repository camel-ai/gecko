from typing import Any, Dict

from fastapi import Request


class RequestDetails:
    """Utility class for extracting and formatting request details."""
    
    @staticmethod
    async def extract(request: Request) -> Dict[str, Any]:
        """Extract all relevant details from a FastAPI request.
        
        Args:
            request: The FastAPI request object
            
        Returns:
            Dictionary containing all request details
        """
        details = {
            'method': request.method,
            'path': str(request.url.path),
            'query_params': dict(request.query_params),
            'headers': dict(request.headers),
            'client': {
                'host': request.client.host if request.client else None,
                'port': request.client.port if request.client else None
            },
            'url': {
                'scheme': request.url.scheme,
                'netloc': request.url.netloc,
                'path': str(request.url.path),
                'query': str(request.url.query),
                'fragment': request.url.fragment
            }
        }
        
        # Try to extract and parse body
        try:
            body = await request.body()
            if body:
                try:
                    details['body'] = body.decode('utf-8')
                except UnicodeDecodeError:
                    details['body'] = "[Binary data]"
        except Exception:
            details['body'] = None
            
        return details
 