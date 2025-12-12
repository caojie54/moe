import io
import sys
import contextlib

def execute_code_with_timeout(code_string, timeout=None):
    """
    高效的内存代码执行，替代原有的进程+文件方案
    - 直接在当前进程中执行代码
    - 捕获stdout输出
    - 移除超时机制以提升性能
    """
    # 创建字符串缓冲区捕获输出
    old_stdout = sys.stdout
    sys.stdout = captured_output = io.StringIO()
    
    try:
        # 创建独立的执行环境，避免变量污染
        exec_globals = {
            '__builtins__': __builtins__,
            # 预导入常用模块，避免重复导入开销
            'typing': __import__('typing'),
            'collections': __import__('collections'), 
            'math': __import__('math'),
        }
        
        # 在隔离环境中执行代码
        exec(code_string, exec_globals, {})
        
        # 获取输出结果
        output = captured_output.getvalue()
        return output
        
    except Exception as e:
        # 将执行错误转换为与原接口兼容的异常
        raise Exception(f"Code execution failed: {str(e)}")
    
    finally:
        # 恢复原始stdout
        sys.stdout = old_stdout
