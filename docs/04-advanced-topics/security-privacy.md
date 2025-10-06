# 🔒 安全与隐私 (Security & Privacy)

## 📖 概述

安全与隐私是nano-vllm在生产环境中的重要考量。本文档详细介绍了身份认证、数据加密、访问控制、隐私保护等安全机制，以及相应的实施策略和最佳实践。

## 🛡️ 安全架构

### 1. 安全管理器

```python
import hashlib
import hmac
import jwt
import bcrypt
import cryptography
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes, serialization
from cryptography.hazmat.primitives.asymmetric import rsa, padding
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
import ssl
import secrets
import time
import logging
import asyncio
import aiohttp
import redis
import sqlite3
import json
import os
import base64
from typing import Dict, List, Optional, Any, Union, Tuple
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import ipaddress
import re
from functools import wraps

logger = logging.getLogger(__name__)

class SecurityLevel(Enum):
    """安全级别"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class AuthMethod(Enum):
    """认证方法"""
    API_KEY = "api_key"
    JWT_TOKEN = "jwt_token"
    OAUTH2 = "oauth2"
    BASIC_AUTH = "basic_auth"
    CERTIFICATE = "certificate"
    MULTI_FACTOR = "multi_factor"

class EncryptionType(Enum):
    """加密类型"""
    AES_256 = "aes_256"
    RSA_2048 = "rsa_2048"
    RSA_4096 = "rsa_4096"
    CHACHA20 = "chacha20"

class AccessLevel(Enum):
    """访问级别"""
    READ_ONLY = "read_only"
    READ_WRITE = "read_write"
    ADMIN = "admin"
    SUPER_ADMIN = "super_admin"

@dataclass
class SecurityConfig:
    """安全配置"""
    security_level: SecurityLevel = SecurityLevel.MEDIUM
    auth_method: AuthMethod = AuthMethod.JWT_TOKEN
    encryption_type: EncryptionType = EncryptionType.AES_256
    
    # JWT配置
    jwt_secret_key: str = ""
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    
    # API Key配置
    api_key_length: int = 32
    api_key_expiration_days: int = 90
    
    # 密码策略
    password_min_length: int = 8
    password_require_uppercase: bool = True
    password_require_lowercase: bool = True
    password_require_numbers: bool = True
    password_require_symbols: bool = True
    
    # 访问控制
    max_login_attempts: int = 5
    lockout_duration_minutes: int = 30
    session_timeout_minutes: int = 60
    
    # 网络安全
    allowed_ips: List[str] = field(default_factory=list)
    blocked_ips: List[str] = field(default_factory=list)
    rate_limit_requests_per_minute: int = 100
    
    # 数据保护
    enable_data_encryption: bool = True
    enable_audit_logging: bool = True
    data_retention_days: int = 365
    
    # SSL/TLS配置
    ssl_cert_path: str = ""
    ssl_key_path: str = ""
    ssl_ca_path: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            'security_level': self.security_level.value,
            'auth_method': self.auth_method.value,
            'encryption_type': self.encryption_type.value,
            'jwt_secret_key': '***' if self.jwt_secret_key else '',
            'jwt_algorithm': self.jwt_algorithm,
            'jwt_expiration_hours': self.jwt_expiration_hours,
            'api_key_length': self.api_key_length,
            'api_key_expiration_days': self.api_key_expiration_days,
            'password_min_length': self.password_min_length,
            'password_require_uppercase': self.password_require_uppercase,
            'password_require_lowercase': self.password_require_lowercase,
            'password_require_numbers': self.password_require_numbers,
            'password_require_symbols': self.password_require_symbols,
            'max_login_attempts': self.max_login_attempts,
            'lockout_duration_minutes': self.lockout_duration_minutes,
            'session_timeout_minutes': self.session_timeout_minutes,
            'allowed_ips': self.allowed_ips,
            'blocked_ips': self.blocked_ips,
            'rate_limit_requests_per_minute': self.rate_limit_requests_per_minute,
            'enable_data_encryption': self.enable_data_encryption,
            'enable_audit_logging': self.enable_audit_logging,
            'data_retention_days': self.data_retention_days,
        }

@dataclass
class User:
    """用户信息"""
    user_id: str
    username: str
    email: str
    password_hash: str
    access_level: AccessLevel
    api_keys: List[str] = field(default_factory=list)
    created_at: datetime = field(default_factory=datetime.now)
    last_login: Optional[datetime] = None
    login_attempts: int = 0
    locked_until: Optional[datetime] = None
    is_active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)

@dataclass
class AuditLog:
    """审计日志"""
    log_id: str
    user_id: str
    action: str
    resource: str
    timestamp: datetime
    ip_address: str
    user_agent: str
    success: bool
    details: Dict[str, Any] = field(default_factory=dict)

class PasswordValidator:
    """密码验证器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
    
    def validate_password(self, password: str) -> Tuple[bool, List[str]]:
        """验证密码强度"""
        
        errors = []
        
        # 检查长度
        if len(password) < self.config.password_min_length:
            errors.append(f"Password must be at least {self.config.password_min_length} characters long")
        
        # 检查大写字母
        if self.config.password_require_uppercase and not re.search(r'[A-Z]', password):
            errors.append("Password must contain at least one uppercase letter")
        
        # 检查小写字母
        if self.config.password_require_lowercase and not re.search(r'[a-z]', password):
            errors.append("Password must contain at least one lowercase letter")
        
        # 检查数字
        if self.config.password_require_numbers and not re.search(r'\d', password):
            errors.append("Password must contain at least one number")
        
        # 检查特殊字符
        if self.config.password_require_symbols and not re.search(r'[!@#$%^&*(),.?":{}|<>]', password):
            errors.append("Password must contain at least one special character")
        
        # 检查常见弱密码
        weak_passwords = [
            'password', '123456', 'qwerty', 'abc123', 'password123',
            'admin', 'root', 'user', 'guest', 'test'
        ]
        
        if password.lower() in weak_passwords:
            errors.append("Password is too common")
        
        return len(errors) == 0, errors
    
    def hash_password(self, password: str) -> str:
        """哈希密码"""
        
        salt = bcrypt.gensalt()
        password_hash = bcrypt.hashpw(password.encode('utf-8'), salt)
        return password_hash.decode('utf-8')
    
    def verify_password(self, password: str, password_hash: str) -> bool:
        """验证密码"""
        
        try:
            return bcrypt.checkpw(password.encode('utf-8'), password_hash.encode('utf-8'))
        except Exception as e:
            logger.error(f"Password verification error: {e}")
            return False

class EncryptionManager:
    """加密管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.fernet_key = None
        self.rsa_private_key = None
        self.rsa_public_key = None
        
        self._initialize_encryption()
    
    def _initialize_encryption(self):
        """初始化加密"""
        
        if self.config.encryption_type == EncryptionType.AES_256:
            self._initialize_aes()
        elif self.config.encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
            self._initialize_rsa()
    
    def _initialize_aes(self):
        """初始化AES加密"""
        
        # 生成或加载Fernet密钥
        key_file = "encryption.key"
        
        if os.path.exists(key_file):
            with open(key_file, 'rb') as f:
                self.fernet_key = f.read()
        else:
            self.fernet_key = Fernet.generate_key()
            with open(key_file, 'wb') as f:
                f.write(self.fernet_key)
    
    def _initialize_rsa(self):
        """初始化RSA加密"""
        
        key_size = 2048 if self.config.encryption_type == EncryptionType.RSA_2048 else 4096
        
        private_key_file = f"rsa_private_{key_size}.pem"
        public_key_file = f"rsa_public_{key_size}.pem"
        
        if os.path.exists(private_key_file) and os.path.exists(public_key_file):
            # 加载现有密钥
            with open(private_key_file, 'rb') as f:
                self.rsa_private_key = serialization.load_pem_private_key(
                    f.read(),
                    password=None
                )
            
            with open(public_key_file, 'rb') as f:
                self.rsa_public_key = serialization.load_pem_public_key(f.read())
        else:
            # 生成新密钥对
            self.rsa_private_key = rsa.generate_private_key(
                public_exponent=65537,
                key_size=key_size
            )
            self.rsa_public_key = self.rsa_private_key.public_key()
            
            # 保存密钥
            with open(private_key_file, 'wb') as f:
                f.write(self.rsa_private_key.private_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PrivateFormat.PKCS8,
                    encryption_algorithm=serialization.NoEncryption()
                ))
            
            with open(public_key_file, 'wb') as f:
                f.write(self.rsa_public_key.public_bytes(
                    encoding=serialization.Encoding.PEM,
                    format=serialization.PublicFormat.SubjectPublicKeyInfo
                ))
    
    def encrypt_data(self, data: str) -> str:
        """加密数据"""
        
        try:
            if self.config.encryption_type == EncryptionType.AES_256:
                return self._encrypt_aes(data)
            elif self.config.encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                return self._encrypt_rsa(data)
            else:
                raise ValueError(f"Unsupported encryption type: {self.config.encryption_type}")
                
        except Exception as e:
            logger.error(f"Encryption error: {e}")
            raise
    
    def decrypt_data(self, encrypted_data: str) -> str:
        """解密数据"""
        
        try:
            if self.config.encryption_type == EncryptionType.AES_256:
                return self._decrypt_aes(encrypted_data)
            elif self.config.encryption_type in [EncryptionType.RSA_2048, EncryptionType.RSA_4096]:
                return self._decrypt_rsa(encrypted_data)
            else:
                raise ValueError(f"Unsupported encryption type: {self.config.encryption_type}")
                
        except Exception as e:
            logger.error(f"Decryption error: {e}")
            raise
    
    def _encrypt_aes(self, data: str) -> str:
        """AES加密"""
        
        fernet = Fernet(self.fernet_key)
        encrypted_data = fernet.encrypt(data.encode('utf-8'))
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def _decrypt_aes(self, encrypted_data: str) -> str:
        """AES解密"""
        
        fernet = Fernet(self.fernet_key)
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = fernet.decrypt(encrypted_bytes)
        return decrypted_data.decode('utf-8')
    
    def _encrypt_rsa(self, data: str) -> str:
        """RSA加密"""
        
        encrypted_data = self.rsa_public_key.encrypt(
            data.encode('utf-8'),
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return base64.b64encode(encrypted_data).decode('utf-8')
    
    def _decrypt_rsa(self, encrypted_data: str) -> str:
        """RSA解密"""
        
        encrypted_bytes = base64.b64decode(encrypted_data.encode('utf-8'))
        decrypted_data = self.rsa_private_key.decrypt(
            encrypted_bytes,
            padding.OAEP(
                mgf=padding.MGF1(algorithm=hashes.SHA256()),
                algorithm=hashes.SHA256(),
                label=None
            )
        )
        return decrypted_data.decode('utf-8')

class AuthenticationManager:
    """认证管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.password_validator = PasswordValidator(config)
        self.users_db = {}  # 简化实现，实际应使用数据库
        self.sessions = {}  # 会话存储
        self.api_keys = {}  # API密钥存储
        
        # 初始化JWT密钥
        if not self.config.jwt_secret_key:
            self.config.jwt_secret_key = secrets.token_urlsafe(32)
    
    async def register_user(self, username: str, email: str, password: str, 
                          access_level: AccessLevel = AccessLevel.READ_ONLY) -> User:
        """注册用户"""
        
        # 验证用户名唯一性
        if any(user.username == username for user in self.users_db.values()):
            raise ValueError("Username already exists")
        
        # 验证邮箱唯一性
        if any(user.email == email for user in self.users_db.values()):
            raise ValueError("Email already exists")
        
        # 验证密码强度
        is_valid, errors = self.password_validator.validate_password(password)
        if not is_valid:
            raise ValueError(f"Password validation failed: {', '.join(errors)}")
        
        # 创建用户
        user_id = secrets.token_urlsafe(16)
        password_hash = self.password_validator.hash_password(password)
        
        user = User(
            user_id=user_id,
            username=username,
            email=email,
            password_hash=password_hash,
            access_level=access_level
        )
        
        self.users_db[user_id] = user
        
        logger.info(f"User registered: {username}")
        
        return user
    
    async def authenticate_user(self, username: str, password: str, 
                              ip_address: str = "") -> Optional[User]:
        """用户认证"""
        
        # 查找用户
        user = None
        for u in self.users_db.values():
            if u.username == username:
                user = u
                break
        
        if not user:
            logger.warning(f"Authentication failed: user not found - {username}")
            return None
        
        # 检查账户锁定
        if user.locked_until and user.locked_until > datetime.now():
            logger.warning(f"Authentication failed: account locked - {username}")
            return None
        
        # 验证密码
        if not self.password_validator.verify_password(password, user.password_hash):
            # 增加登录失败次数
            user.login_attempts += 1
            
            if user.login_attempts >= self.config.max_login_attempts:
                user.locked_until = datetime.now() + timedelta(
                    minutes=self.config.lockout_duration_minutes
                )
                logger.warning(f"Account locked due to too many failed attempts: {username}")
            
            logger.warning(f"Authentication failed: invalid password - {username}")
            return None
        
        # 重置登录失败次数
        user.login_attempts = 0
        user.locked_until = None
        user.last_login = datetime.now()
        
        logger.info(f"User authenticated successfully: {username}")
        
        return user
    
    def generate_jwt_token(self, user: User) -> str:
        """生成JWT令牌"""
        
        payload = {
            'user_id': user.user_id,
            'username': user.username,
            'access_level': user.access_level.value,
            'exp': datetime.utcnow() + timedelta(hours=self.config.jwt_expiration_hours),
            'iat': datetime.utcnow()
        }
        
        token = jwt.encode(
            payload,
            self.config.jwt_secret_key,
            algorithm=self.config.jwt_algorithm
        )
        
        return token
    
    def verify_jwt_token(self, token: str) -> Optional[Dict[str, Any]]:
        """验证JWT令牌"""
        
        try:
            payload = jwt.decode(
                token,
                self.config.jwt_secret_key,
                algorithms=[self.config.jwt_algorithm]
            )
            
            # 检查用户是否仍然存在且活跃
            user = self.users_db.get(payload['user_id'])
            if not user or not user.is_active:
                return None
            
            return payload
            
        except jwt.ExpiredSignatureError:
            logger.warning("JWT token expired")
            return None
        except jwt.InvalidTokenError as e:
            logger.warning(f"Invalid JWT token: {e}")
            return None
    
    def generate_api_key(self, user: User) -> str:
        """生成API密钥"""
        
        api_key = secrets.token_urlsafe(self.config.api_key_length)
        
        # 存储API密钥信息
        self.api_keys[api_key] = {
            'user_id': user.user_id,
            'created_at': datetime.now(),
            'expires_at': datetime.now() + timedelta(days=self.config.api_key_expiration_days),
            'is_active': True
        }
        
        # 添加到用户的API密钥列表
        user.api_keys.append(api_key)
        
        logger.info(f"API key generated for user: {user.username}")
        
        return api_key
    
    def verify_api_key(self, api_key: str) -> Optional[User]:
        """验证API密钥"""
        
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return None
        
        # 检查密钥是否过期
        if key_info['expires_at'] < datetime.now():
            logger.warning(f"API key expired: {api_key[:8]}...")
            return None
        
        # 检查密钥是否活跃
        if not key_info['is_active']:
            return None
        
        # 获取用户信息
        user = self.users_db.get(key_info['user_id'])
        if not user or not user.is_active:
            return None
        
        return user
    
    def revoke_api_key(self, api_key: str) -> bool:
        """撤销API密钥"""
        
        key_info = self.api_keys.get(api_key)
        if not key_info:
            return False
        
        key_info['is_active'] = False
        
        # 从用户的API密钥列表中移除
        user = self.users_db.get(key_info['user_id'])
        if user and api_key in user.api_keys:
            user.api_keys.remove(api_key)
        
        logger.info(f"API key revoked: {api_key[:8]}...")
        
        return True

class AccessControlManager:
    """访问控制管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.rate_limiter = {}  # 速率限制器
        self.ip_whitelist = set(config.allowed_ips)
        self.ip_blacklist = set(config.blocked_ips)
    
    def check_ip_access(self, ip_address: str) -> bool:
        """检查IP访问权限"""
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            # 检查黑名单
            for blocked_ip in self.ip_blacklist:
                if ip in ipaddress.ip_network(blocked_ip, strict=False):
                    logger.warning(f"IP blocked: {ip_address}")
                    return False
            
            # 检查白名单（如果配置了白名单）
            if self.ip_whitelist:
                for allowed_ip in self.ip_whitelist:
                    if ip in ipaddress.ip_network(allowed_ip, strict=False):
                        return True
                
                logger.warning(f"IP not in whitelist: {ip_address}")
                return False
            
            return True
            
        except ValueError:
            logger.error(f"Invalid IP address: {ip_address}")
            return False
    
    def check_rate_limit(self, identifier: str, ip_address: str = "") -> bool:
        """检查速率限制"""
        
        current_time = time.time()
        window_start = current_time - 60  # 1分钟窗口
        
        # 使用IP地址或标识符作为限制键
        limit_key = ip_address or identifier
        
        if limit_key not in self.rate_limiter:
            self.rate_limiter[limit_key] = []
        
        # 清理过期的请求记录
        self.rate_limiter[limit_key] = [
            timestamp for timestamp in self.rate_limiter[limit_key]
            if timestamp > window_start
        ]
        
        # 检查是否超过限制
        if len(self.rate_limiter[limit_key]) >= self.config.rate_limit_requests_per_minute:
            logger.warning(f"Rate limit exceeded for {limit_key}")
            return False
        
        # 记录当前请求
        self.rate_limiter[limit_key].append(current_time)
        
        return True
    
    def check_access_permission(self, user: User, resource: str, action: str) -> bool:
        """检查访问权限"""
        
        # 基于用户访问级别的权限检查
        if user.access_level == AccessLevel.SUPER_ADMIN:
            return True
        
        if user.access_level == AccessLevel.ADMIN:
            # 管理员可以访问大部分资源
            restricted_actions = ['delete_user', 'modify_security_config']
            return action not in restricted_actions
        
        if user.access_level == AccessLevel.READ_WRITE:
            # 读写用户可以进行读写操作
            allowed_actions = ['read', 'write', 'generate', 'inference']
            return action in allowed_actions
        
        if user.access_level == AccessLevel.READ_ONLY:
            # 只读用户只能进行读操作
            allowed_actions = ['read', 'view', 'list']
            return action in allowed_actions
        
        return False
    
    def add_ip_to_blacklist(self, ip_address: str):
        """添加IP到黑名单"""
        
        self.ip_blacklist.add(ip_address)
        logger.info(f"IP added to blacklist: {ip_address}")
    
    def remove_ip_from_blacklist(self, ip_address: str):
        """从黑名单移除IP"""
        
        self.ip_blacklist.discard(ip_address)
        logger.info(f"IP removed from blacklist: {ip_address}")
    
    def add_ip_to_whitelist(self, ip_address: str):
        """添加IP到白名单"""
        
        self.ip_whitelist.add(ip_address)
        logger.info(f"IP added to whitelist: {ip_address}")
    
    def remove_ip_from_whitelist(self, ip_address: str):
        """从白名单移除IP"""
        
        self.ip_whitelist.discard(ip_address)
        logger.info(f"IP removed from whitelist: {ip_address}")

class AuditLogger:
    """审计日志记录器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.logs = []  # 简化实现，实际应使用数据库
        self.encryption_manager = None
        
        if config.enable_data_encryption:
            self.encryption_manager = EncryptionManager(config)
    
    async def log_action(self, user_id: str, action: str, resource: str,
                        ip_address: str = "", user_agent: str = "",
                        success: bool = True, details: Dict[str, Any] = None):
        """记录操作日志"""
        
        if not self.config.enable_audit_logging:
            return
        
        log_entry = AuditLog(
            log_id=secrets.token_urlsafe(16),
            user_id=user_id,
            action=action,
            resource=resource,
            timestamp=datetime.now(),
            ip_address=ip_address,
            user_agent=user_agent,
            success=success,
            details=details or {}
        )
        
        # 加密敏感信息
        if self.encryption_manager and self.config.enable_data_encryption:
            if 'password' in log_entry.details:
                log_entry.details['password'] = '***'
            if 'token' in log_entry.details:
                log_entry.details['token'] = self.encryption_manager.encrypt_data(
                    log_entry.details['token']
                )
        
        self.logs.append(log_entry)
        
        # 清理过期日志
        await self._cleanup_old_logs()
        
        logger.info(f"Audit log recorded: {action} on {resource} by {user_id}")
    
    async def _cleanup_old_logs(self):
        """清理过期日志"""
        
        cutoff_date = datetime.now() - timedelta(days=self.config.data_retention_days)
        
        original_count = len(self.logs)
        self.logs = [log for log in self.logs if log.timestamp > cutoff_date]
        
        cleaned_count = original_count - len(self.logs)
        if cleaned_count > 0:
            logger.info(f"Cleaned up {cleaned_count} old audit logs")
    
    def get_logs(self, user_id: str = "", action: str = "", 
                start_time: datetime = None, end_time: datetime = None,
                limit: int = 100) -> List[AuditLog]:
        """获取审计日志"""
        
        filtered_logs = self.logs
        
        # 按用户ID过滤
        if user_id:
            filtered_logs = [log for log in filtered_logs if log.user_id == user_id]
        
        # 按操作过滤
        if action:
            filtered_logs = [log for log in filtered_logs if log.action == action]
        
        # 按时间范围过滤
        if start_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp >= start_time]
        
        if end_time:
            filtered_logs = [log for log in filtered_logs if log.timestamp <= end_time]
        
        # 按时间倒序排列并限制数量
        filtered_logs.sort(key=lambda x: x.timestamp, reverse=True)
        
        return filtered_logs[:limit]
    
    def get_security_events(self, severity: str = "high") -> List[AuditLog]:
        """获取安全事件"""
        
        security_actions = [
            'login_failed', 'account_locked', 'unauthorized_access',
            'permission_denied', 'rate_limit_exceeded', 'suspicious_activity'
        ]
        
        security_logs = [
            log for log in self.logs
            if log.action in security_actions or not log.success
        ]
        
        return security_logs

class PrivacyManager:
    """隐私管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.encryption_manager = EncryptionManager(config)
        self.data_anonymizer = DataAnonymizer()
    
    def anonymize_user_data(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """匿名化用户数据"""
        
        return self.data_anonymizer.anonymize(data)
    
    def encrypt_sensitive_data(self, data: str) -> str:
        """加密敏感数据"""
        
        return self.encryption_manager.encrypt_data(data)
    
    def decrypt_sensitive_data(self, encrypted_data: str) -> str:
        """解密敏感数据"""
        
        return self.encryption_manager.decrypt_data(encrypted_data)
    
    def mask_sensitive_fields(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """掩码敏感字段"""
        
        sensitive_fields = [
            'password', 'token', 'api_key', 'secret', 'private_key',
            'ssn', 'credit_card', 'phone', 'email'
        ]
        
        masked_data = data.copy()
        
        for field in sensitive_fields:
            if field in masked_data:
                value = str(masked_data[field])
                if len(value) > 4:
                    masked_data[field] = value[:2] + '*' * (len(value) - 4) + value[-2:]
                else:
                    masked_data[field] = '*' * len(value)
        
        return masked_data
    
    def generate_privacy_report(self, user_id: str) -> Dict[str, Any]:
        """生成隐私报告"""
        
        return {
            'user_id': user_id,
            'data_collected': [
                'username', 'email', 'login_history', 'api_usage'
            ],
            'data_usage_purpose': [
                'authentication', 'authorization', 'audit_logging', 'service_improvement'
            ],
            'data_retention_period': f"{self.config.data_retention_days} days",
            'data_sharing': 'No data sharing with third parties',
            'user_rights': [
                'access_data', 'correct_data', 'delete_data', 'data_portability'
            ],
            'generated_at': datetime.now().isoformat()
        }

class DataAnonymizer:
    """数据匿名化器"""
    
    def __init__(self):
        self.anonymization_rules = {
            'email': self._anonymize_email,
            'phone': self._anonymize_phone,
            'ip_address': self._anonymize_ip,
            'user_agent': self._anonymize_user_agent,
            'name': self._anonymize_name
        }
    
    def anonymize(self, data: Dict[str, Any]) -> Dict[str, Any]:
        """匿名化数据"""
        
        anonymized_data = {}
        
        for key, value in data.items():
            if key in self.anonymization_rules:
                anonymized_data[key] = self.anonymization_rules[key](value)
            else:
                anonymized_data[key] = value
        
        return anonymized_data
    
    def _anonymize_email(self, email: str) -> str:
        """匿名化邮箱"""
        
        if '@' not in email:
            return email
        
        local, domain = email.split('@', 1)
        
        if len(local) <= 2:
            anonymized_local = '*' * len(local)
        else:
            anonymized_local = local[0] + '*' * (len(local) - 2) + local[-1]
        
        return f"{anonymized_local}@{domain}"
    
    def _anonymize_phone(self, phone: str) -> str:
        """匿名化电话号码"""
        
        if len(phone) <= 4:
            return '*' * len(phone)
        
        return phone[:2] + '*' * (len(phone) - 4) + phone[-2:]
    
    def _anonymize_ip(self, ip_address: str) -> str:
        """匿名化IP地址"""
        
        try:
            ip = ipaddress.ip_address(ip_address)
            
            if ip.version == 4:
                # IPv4: 保留前两个八位组
                parts = ip_address.split('.')
                return f"{parts[0]}.{parts[1]}.*.* "
            else:
                # IPv6: 保留前64位
                return ip_address[:19] + '::*'
                
        except ValueError:
            return ip_address
    
    def _anonymize_user_agent(self, user_agent: str) -> str:
        """匿名化User Agent"""
        
        # 保留浏览器类型，移除版本信息
        patterns = [
            (r'Chrome/[\d.]+', 'Chrome/*'),
            (r'Firefox/[\d.]+', 'Firefox/*'),
            (r'Safari/[\d.]+', 'Safari/*'),
            (r'Edge/[\d.]+', 'Edge/*')
        ]
        
        anonymized_ua = user_agent
        for pattern, replacement in patterns:
            anonymized_ua = re.sub(pattern, replacement, anonymized_ua)
        
        return anonymized_ua
    
    def _anonymize_name(self, name: str) -> str:
        """匿名化姓名"""
        
        if len(name) <= 2:
            return '*' * len(name)
        
        return name[0] + '*' * (len(name) - 2) + name[-1]

class SecurityManager:
    """安全管理器"""
    
    def __init__(self, config: SecurityConfig):
        self.config = config
        self.auth_manager = AuthenticationManager(config)
        self.access_control = AccessControlManager(config)
        self.audit_logger = AuditLogger(config)
        self.privacy_manager = PrivacyManager(config)
        self.encryption_manager = EncryptionManager(config)
        
    async def authenticate_request(self, request_data: Dict[str, Any]) -> Optional[User]:
        """认证请求"""
        
        auth_header = request_data.get('authorization', '')
        ip_address = request_data.get('ip_address', '')
        user_agent = request_data.get('user_agent', '')
        
        # 检查IP访问权限
        if not self.access_control.check_ip_access(ip_address):
            await self.audit_logger.log_action(
                user_id='unknown',
                action='ip_blocked',
                resource='authentication',
                ip_address=ip_address,
                user_agent=user_agent,
                success=False
            )
            return None
        
        user = None
        
        try:
            if auth_header.startswith('Bearer '):
                # JWT令牌认证
                token = auth_header[7:]
                payload = self.auth_manager.verify_jwt_token(token)
                
                if payload:
                    user = self.auth_manager.users_db.get(payload['user_id'])
                    
                    await self.audit_logger.log_action(
                        user_id=payload['user_id'],
                        action='jwt_authentication',
                        resource='authentication',
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=True
                    )
            
            elif auth_header.startswith('ApiKey '):
                # API密钥认证
                api_key = auth_header[7:]
                user = self.auth_manager.verify_api_key(api_key)
                
                if user:
                    await self.audit_logger.log_action(
                        user_id=user.user_id,
                        action='api_key_authentication',
                        resource='authentication',
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=True
                    )
            
            elif auth_header.startswith('Basic '):
                # 基础认证
                credentials = base64.b64decode(auth_header[6:]).decode('utf-8')
                username, password = credentials.split(':', 1)
                
                user = await self.auth_manager.authenticate_user(
                    username, password, ip_address
                )
                
                await self.audit_logger.log_action(
                    user_id=user.user_id if user else 'unknown',
                    action='basic_authentication',
                    resource='authentication',
                    ip_address=ip_address,
                    user_agent=user_agent,
                    success=user is not None
                )
            
            # 检查速率限制
            if user:
                if not self.access_control.check_rate_limit(user.user_id, ip_address):
                    await self.audit_logger.log_action(
                        user_id=user.user_id,
                        action='rate_limit_exceeded',
                        resource='authentication',
                        ip_address=ip_address,
                        user_agent=user_agent,
                        success=False
                    )
                    return None
            
            return user
            
        except Exception as e:
            logger.error(f"Authentication error: {e}")
            
            await self.audit_logger.log_action(
                user_id='unknown',
                action='authentication_error',
                resource='authentication',
                ip_address=ip_address,
                user_agent=user_agent,
                success=False,
                details={'error': str(e)}
            )
            
            return None
    
    async def authorize_request(self, user: User, resource: str, action: str,
                              request_data: Dict[str, Any]) -> bool:
        """授权请求"""
        
        ip_address = request_data.get('ip_address', '')
        user_agent = request_data.get('user_agent', '')
        
        # 检查访问权限
        has_permission = self.access_control.check_access_permission(user, resource, action)
        
        await self.audit_logger.log_action(
            user_id=user.user_id,
            action=f'authorize_{action}',
            resource=resource,
            ip_address=ip_address,
            user_agent=user_agent,
            success=has_permission
        )
        
        return has_permission
    
    def secure_middleware(self, handler):
        """安全中间件装饰器"""
        
        @wraps(handler)
        async def wrapper(*args, **kwargs):
            # 提取请求信息
            request_data = kwargs.get('request_data', {})
            
            # 认证
            user = await self.authenticate_request(request_data)
            if not user:
                return {'error': 'Authentication failed', 'status': 401}
            
            # 授权
            resource = kwargs.get('resource', 'unknown')
            action = kwargs.get('action', 'unknown')
            
            if not await self.authorize_request(user, resource, action, request_data):
                return {'error': 'Authorization failed', 'status': 403}
            
            # 添加用户信息到请求
            kwargs['user'] = user
            
            try:
                # 执行处理函数
                result = await handler(*args, **kwargs)
                
                # 记录成功操作
                await self.audit_logger.log_action(
                    user_id=user.user_id,
                    action=action,
                    resource=resource,
                    ip_address=request_data.get('ip_address', ''),
                    user_agent=request_data.get('user_agent', ''),
                    success=True
                )
                
                return result
                
            except Exception as e:
                # 记录失败操作
                await self.audit_logger.log_action(
                    user_id=user.user_id,
                    action=action,
                    resource=resource,
                    ip_address=request_data.get('ip_address', ''),
                    user_agent=request_data.get('user_agent', ''),
                    success=False,
                    details={'error': str(e)}
                )
                
                raise
        
        return wrapper
    
    async def get_security_status(self) -> Dict[str, Any]:
        """获取安全状态"""
        
        return {
            'config': self.config.to_dict(),
            'users_count': len(self.auth_manager.users_db),
            'active_sessions': len(self.auth_manager.sessions),
            'api_keys_count': len(self.auth_manager.api_keys),
            'audit_logs_count': len(self.audit_logger.logs),
            'ip_whitelist_count': len(self.access_control.ip_whitelist),
            'ip_blacklist_count': len(self.access_control.ip_blacklist),
            'security_level': self.config.security_level.value,
            'encryption_enabled': self.config.enable_data_encryption,
            'audit_logging_enabled': self.config.enable_audit_logging
        }
    
    async def generate_security_report(self) -> Dict[str, Any]:
        """生成安全报告"""
        
        # 获取最近的安全事件
        security_events = self.audit_logger.get_security_events()
        
        # 统计信息
        failed_logins = len([log for log in security_events if log.action == 'login_failed'])
        blocked_ips = len([log for log in security_events if log.action == 'ip_blocked'])
        rate_limit_violations = len([log for log in security_events if log.action == 'rate_limit_exceeded'])
        
        return {
            'report_generated_at': datetime.now().isoformat(),
            'security_level': self.config.security_level.value,
            'total_users': len(self.auth_manager.users_db),
            'active_users': len([u for u in self.auth_manager.users_db.values() if u.is_active]),
            'locked_users': len([u for u in self.auth_manager.users_db.values() if u.locked_until]),
            'total_api_keys': len(self.auth_manager.api_keys),
            'active_api_keys': len([k for k in self.auth_manager.api_keys.values() if k['is_active']]),
            'security_events': {
                'failed_logins': failed_logins,
                'blocked_ips': blocked_ips,
                'rate_limit_violations': rate_limit_violations,
                'total_events': len(security_events)
            },
            'ip_access_control': {
                'whitelist_count': len(self.access_control.ip_whitelist),
                'blacklist_count': len(self.access_control.ip_blacklist)
            },
            'encryption_status': {
                'enabled': self.config.enable_data_encryption,
                'type': self.config.encryption_type.value
            },
            'audit_logging': {
                'enabled': self.config.enable_audit_logging,
                'total_logs': len(self.audit_logger.logs),
                'retention_days': self.config.data_retention_days
            }
        }
```

## 🚀 使用示例

### 1. 基础安全配置

```python
async def basic_security_example():
    # 创建安全配置
    config = SecurityConfig(
        security_level=SecurityLevel.HIGH,
        auth_method=AuthMethod.JWT_TOKEN,
        encryption_type=EncryptionType.AES_256,
        jwt_expiration_hours=24,
        password_min_length=12,
        max_login_attempts=3,
        lockout_duration_minutes=15,
        rate_limit_requests_per_minute=60,
        enable_data_encryption=True,
        enable_audit_logging=True
    )
    
    # 创建安全管理器
    security_manager = SecurityManager(config)
    
    # 注册用户
    user = await security_manager.auth_manager.register_user(
        username="testuser",
        email="test@example.com",
        password="SecurePassword123!",
        access_level=AccessLevel.READ_WRITE
    )
    
    print(f"User registered: {user.username}")
    
    # 生成JWT令牌
    token = security_manager.auth_manager.generate_jwt_token(user)
    print(f"JWT token generated: {token[:20]}...")
    
    # 生成API密钥
    api_key = security_manager.auth_manager.generate_api_key(user)
    print(f"API key generated: {api_key[:8]}...")
    
    # 模拟请求认证
    request_data = {
        'authorization': f'Bearer {token}',
        'ip_address': '192.168.1.100',
        'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    authenticated_user = await security_manager.authenticate_request(request_data)
    
    if authenticated_user:
        print(f"Authentication successful: {authenticated_user.username}")
        
        # 检查授权
        authorized = await security_manager.authorize_request(
            authenticated_user,
            resource="model_inference",
            action="generate",
            request_data=request_data
        )
        
        print(f"Authorization result: {authorized}")
    
    # 获取安全状态
    status = await security_manager.get_security_status()
    print(f"Security status: {status}")

if __name__ == "__main__":
    asyncio.run(basic_security_example())
```

### 2. 数据加密示例

```python
async def encryption_example():
    # 创建加密配置
    config = SecurityConfig(
        encryption_type=EncryptionType.AES_256,
        enable_data_encryption=True
    )
    
    # 创建加密管理器
    encryption_manager = EncryptionManager(config)
    
    # 加密敏感数据
    sensitive_data = "This is sensitive information that needs to be encrypted"
    
    encrypted_data = encryption_manager.encrypt_data(sensitive_data)
    print(f"Encrypted data: {encrypted_data[:50]}...")
    
    # 解密数据
    decrypted_data = encryption_manager.decrypt_data(encrypted_data)
    print(f"Decrypted data: {decrypted_data}")
    
    # 验证数据完整性
    assert sensitive_data == decrypted_data
    print("Encryption/decryption successful!")
    
    # 隐私管理示例
    privacy_manager = PrivacyManager(config)
    
    # 匿名化用户数据
    user_data = {
        'email': 'user@example.com',
        'phone': '+1234567890',
        'ip_address': '192.168.1.100',
        'name': 'John Doe'
    }
    
    anonymized_data = privacy_manager.anonymize_user_data(user_data)
    print(f"Anonymized data: {anonymized_data}")
    
    # 掩码敏感字段
    sensitive_fields_data = {
        'username': 'testuser',
        'password': 'secret123',
        'api_key': 'sk-1234567890abcdef',
        'email': 'user@example.com'
    }
    
    masked_data = privacy_manager.mask_sensitive_fields(sensitive_fields_data)
    print(f"Masked data: {masked_data}")

if __name__ == "__main__":
    asyncio.run(encryption_example())
```

### 3. 安全中间件示例

```python
async def security_middleware_example():
    # 创建安全管理器
    config = SecurityConfig(
        security_level=SecurityLevel.HIGH,
        auth_method=AuthMethod.JWT_TOKEN
    )
    
    security_manager = SecurityManager(config)
    
    # 注册测试用户
    user = await security_manager.auth_manager.register_user(
        username="apiuser",
        email="api@example.com",
        password="ApiPassword123!",
        access_level=AccessLevel.READ_WRITE
    )
    
    token = security_manager.auth_manager.generate_jwt_token(user)
    
    # 使用安全中间件装饰的处理函数
    @security_manager.secure_middleware
    async def protected_inference_handler(request_data, user, resource, action):
        """受保护的推理处理函数"""
        
        prompt = request_data.get('prompt', '')
        
        # 模拟推理逻辑
        result = {
            'user_id': user.user_id,
            'prompt': prompt,
            'response': f"Generated response for: {prompt}",
            'timestamp': datetime.now().isoformat()
        }
        
        return result
    
    # 模拟API请求
    request_data = {
        'authorization': f'Bearer {token}',
        'ip_address': '192.168.1.100',
        'user_agent': 'API Client/1.0',
        'prompt': 'What is artificial intelligence?'
    }
    
    try:
        # 调用受保护的处理函数
        result = await protected_inference_handler(
            request_data=request_data,
            resource="model_inference",
            action="generate"
        )
        
        print(f"Inference result: {result}")
        
    except Exception as e:
        print(f"Error: {e}")
    
    # 生成安全报告
    security_report = await security_manager.generate_security_report()
    print(f"Security report: {json.dumps(security_report, indent=2)}")

if __name__ == "__main__":
    asyncio.run(security_middleware_example())
```

### 4. 审计日志示例

```python
async def audit_logging_example():
    # 创建安全配置
    config = SecurityConfig(
        enable_audit_logging=True,
        data_retention_days=30
    )
    
    # 创建审计日志记录器
    audit_logger = AuditLogger(config)
    
    # 记录各种操作
    await audit_logger.log_action(
        user_id="user123",
        action="login",
        resource="authentication",
        ip_address="192.168.1.100",
        user_agent="Mozilla/5.0",
        success=True
    )
    
    await audit_logger.log_action(
        user_id="user123",
        action="model_inference",
        resource="llama-7b",
        ip_address="192.168.1.100",
        success=True,
        details={'prompt_length': 50, 'response_length': 200}
    )
    
    await audit_logger.log_action(
        user_id="user456",
        action="login",
        resource="authentication",
        ip_address="10.0.0.50",
        success=False,
        details={'reason': 'invalid_password'}
    )
    
    # 查询审计日志
    all_logs = audit_logger.get_logs(limit=10)
    print(f"Total logs: {len(all_logs)}")
    
    # 查询特定用户的日志
    user_logs = audit_logger.get_logs(user_id="user123")
    print(f"User123 logs: {len(user_logs)}")
    
    # 查询安全事件
    security_events = audit_logger.get_security_events()
    print(f"Security events: {len(security_events)}")
    
    # 打印日志详情
    for log in all_logs:
        print(f"[{log.timestamp}] {log.user_id} - {log.action} on {log.resource} - Success: {log.success}")

if __name__ == "__main__":
    asyncio.run(audit_logging_example())
```

## 🎯 最佳实践

### 1. 认证策略
- **多因素认证**: 对高权限用户启用MFA
- **令牌管理**: 定期轮换JWT密钥和API密钥
- **会话管理**: 设置合理的会话超时时间
- **密码策略**: 强制使用强密码并定期更换

### 2. 访问控制
- **最小权限原则**: 只授予必要的最小权限
- **角色分离**: 明确区分不同角色的权限
- **IP白名单**: 限制可访问的IP地址范围
- **速率限制**: 防止API滥用和DDoS攻击

### 3. 数据保护
- **传输加密**: 使用HTTPS/TLS加密数据传输
- **存储加密**: 加密存储敏感数据
- **数据匿名化**: 对日志和分析数据进行匿名化
- **数据备份**: 定期备份并测试恢复流程

### 4. 监控告警
- **实时监控**: 监控异常登录和访问行为
- **安全事件**: 及时响应安全事件和威胁
- **审计日志**: 完整记录所有关键操作
- **合规检查**: 定期进行安全合规检查

## 📈 总结

安全与隐私是nano-vllm生产部署的重要基础。通过实施全面的安全策略和隐私保护措施，可以确保系统的安全性和用户数据的隐私性。

关键要点：
1. **多层防护**: 实施认证、授权、加密、审计等多层安全防护
2. **隐私保护**: 采用数据匿名化、加密存储等隐私保护技术
3. **持续监控**: 建立完善的安全监控和事件响应机制
4. **合规管理**: 遵循相关法规和行业标准

通过遵循安全最佳实践和持续改进，可以构建安全可靠的nano-vllm服务环境。