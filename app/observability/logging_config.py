import logging
import json
from datetime import datetime

class JSONFormatter(logging.Formatter):
    def format(self, record):
        log_data = {
            "timestamp": datetime.utcnow().isoformat() + "Z",
            "levelname": record.levelname,
            "message": record.getMessage(),
            "name": record.name,
        }
        
        # Agregar contexto de request si está disponible
        if hasattr(record, 'endpoint'):
            log_data['endpoint'] = record.endpoint
        if hasattr(record, 'ip'):
            log_data['ip'] = record.ip
        if hasattr(record, 'status'):
            log_data['status'] = record.status
        if hasattr(record, 'method'):
            log_data['method'] = record.method
        
        # Agregar detalles de excepciones
        if record.exc_info:
            log_data['error'] = True
            log_data['exception'] = self.formatException(record.exc_info)
            log_data['type'] = record.exc_info[0].__name__ if record.exc_info[0] else None
        else:
            log_data['error'] = False
        
        return json.dumps(log_data, ensure_ascii=False)

def setup_logger(name="app"):
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)
    
    # Handler para consola
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.DEBUG)
    console_handler.setFormatter(JSONFormatter())
    
    logger.addHandler(console_handler)
    
    return logger
