from fastapi import Request, HTTPException, status
from typing import Optional, Dict, Any, Tuple
import logging

logger = logging.getLogger(__name__)

def get_authenticated_context(request: Request) -> Tuple[Optional[Any], Dict[str, Any], Optional[str]]:
    """
    Helper function to extract authenticated context from request
    
    Returns:
        Tuple of (authenticated_user, session_context, chat_session_id)
    """
    try:
        authenticated_user = getattr(request.state, 'authenticated_user', None)
        session_context = getattr(request.state, 'session_context', {})
        chat_session_id = getattr(request.state, 'chat_session_id', None)
        
        return authenticated_user, session_context, chat_session_id
    except Exception as e:
        logger.error(f"Failed to get authenticated context: {e}")
        return None, {}, None

def require_authenticated_user(request: Request) -> Any:
    """
    Helper function to ensure user is authenticated
    
    Raises HTTPException if user is not authenticated
    Returns the authenticated user object
    """
    authenticated_user, _, _ = get_authenticated_context(request)
    
    if not authenticated_user:
        logger.error("Request attempted without authentication")
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Authentication required but not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    
    return authenticated_user

def get_user_namespace(request: Request, doc_id: str) -> str:
    """
    Helper function to construct user namespace safely
    
    Args:
        request: FastAPI request object with authenticated context
        doc_id: Document ID
    
    Returns:
        Namespace string in format: user_{user_id}_doc_{doc_id}
    """
    authenticated_user = require_authenticated_user(request)
    return f"user_{authenticated_user.id}_doc_{doc_id}"

def enhance_arguments_with_context(
    request: Request, 
    arguments: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Helper function to enhance MCP tool arguments with authenticated context
    
    Args:
        request: FastAPI request object
        arguments: Original tool arguments
    
    Returns:
        Enhanced arguments with user context
    """
    authenticated_user, session_context, chat_session_id = get_authenticated_context(request)
    
    enhanced_arguments = arguments.copy()
    
    # Add user context
    if authenticated_user:
        enhanced_arguments['_user_id'] = authenticated_user.id
        enhanced_arguments['_user_email'] = authenticated_user.email
        enhanced_arguments['_user_role'] = authenticated_user.role  # NEW: Include user role
    
    # Add session context
    if chat_session_id:
        enhanced_arguments['_session_id'] = chat_session_id
    
    # Add namespace if available
    if session_context.get('namespace'):
        enhanced_arguments['_namespace'] = session_context['namespace']
    
    # Add other session metadata
    if session_context.get('session_mode'):
        enhanced_arguments['_session_mode'] = session_context['session_mode']
    
    return enhanced_arguments

def log_authenticated_request(
    request: Request, 
    endpoint_name: str, 
    additional_info: Optional[Dict[str, Any]] = None
):
    """
    Helper function to log authenticated requests consistently
    
    Args:
        request: FastAPI request object
        endpoint_name: Name of the endpoint being called
        additional_info: Additional information to log
    """
    authenticated_user, session_context, chat_session_id = get_authenticated_context(request)
    
    logger.info(f"🎯 {endpoint_name}")
    logger.info(f"👤 User: {authenticated_user.email if authenticated_user else 'Unknown'}")
    logger.info(f"🔐 Session: {chat_session_id or 'None'}")
    logger.info(f"📁 Namespace: {session_context.get('namespace', 'None')}")
    
    if additional_info:
        for key, value in additional_info.items():
            logger.info(f"�� {key}: {value}") 