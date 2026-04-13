import pytest
import os
from unittest.mock import patch, MagicMock
from config.settings import HF_API_TOKEN

# We mock HF_API_TOKEN to ensure the tool tries to hit the logic.
@patch.dict(os.environ, {"HF_API_TOKEN": "mock_token"})
@pytest.mark.asyncio
async def test_image_tool():
    from tools.image_tool import generate_image_tool
    
    with patch("tools.image_tool.requests.post") as mock_post:
        # Create a mock response
        mock_resp = MagicMock()
        mock_resp.status_code = 200
        mock_resp.headers = {"content-type": "image/png"}
        
        # Create a tiny valid dummy PNG byte stream
        from PIL import Image
        import io
        img = Image.new('RGB', (10, 10))
        buf = io.BytesIO()
        img.save(buf, format='PNG')
        mock_resp.content = buf.getvalue()
        
        mock_post.return_value = mock_resp
        
        # Test the tool execution
        result = await generate_image_tool._arun(prompt="test", output_folder="tests/tmp")
        
        assert "illustration" in result
        assert ".png" in result

@pytest.mark.asyncio
async def test_pdf_tool():
    from tools.pdf_tool import generate_pdf_tool
    
    # We pass a sufficiently long string (200+ chars)
    dummy_text = "This is a test document. " * 20
    
    result = await generate_pdf_tool._arun(
        title="Test Report", 
        body_text=dummy_text,
        output_folder="tests/tmp",
        image_path="",
        target_group="adults",
        language="en",
        accessibility_score=5.0
    )
    
    assert "report" in result.lower() or "test" in result.lower()
    assert ".pdf" in result
