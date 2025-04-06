from IPython.display import display, HTML
import io
import base64

# hiển thị dưới dạng scroll view html

def html_view(df=None):
    if df is None:
        return
    
    html = df.to_html(index=False)
    scroll_html = f"""
    <style>
        .scroll-table {{
            max-height: 400px;
            overflow: auto;
            border: 1px solid #ddd;
            position: relative;
        }}
        .scroll-table table {{
            width: 100%;
            border-collapse: collapse;
            table-layout: auto;
        }}
        .scroll-table thead {{
            position: sticky;
            top: 0;
            background: #f8f9fa;
            color: black;
            z-index: 2;
        }}
        .scroll-table th, .scroll-table td {{
            padding: 8px;
            border: 1px solid #ddd;
            text-align: left;
            white-space: normal;
            word-wrap: break-word;
        }}
    </style>
    <div class="scroll-table">
        {html}
    </div>
    """
    
    display(HTML(scroll_html))
    
def html_view_image(image_buffer):
    img_str = base64.b64encode(image_buffer.getvalue()).decode()
    html = f'<div style="width:100%; max-height:800px; overflow:auto;"><img src="data:image/png;base64,{img_str}" /></div>'
    display(HTML(html))