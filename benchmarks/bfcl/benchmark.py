"""
BFCL Benchmark Implementation
BFCL (Berkeley Function Call Leaderboard) benchmark plugin
"""

import logging
from typing import Any, Dict, List, Optional

from ..base.benchmark import BaseBenchmark
from ..base.test_case import TestCase
from ..base.execution_result import ExecutionResult
from .evaluation.evaluator import BFCLEvaluator
from .loader import BFCLTestLoader
from .utils import derive_single_turn_runtime_function_name
from utils.test_case_adapter import TestCaseAdapter

logger = logging.getLogger(__name__)

# Multi-turn function name mapping table
# Maps from agent-generated names to expected ground truth names
MULTI_TURN_FUNCTION_MAPPING = {
    # GorillaFileSystem mappings (still lowercase internally)
    'gorillafilesystem_cat': 'cat',
    'gorillafilesystem_cd': 'cd',
    'gorillafilesystem_cp': 'cp',
    'gorillafilesystem_diff': 'diff',
    'gorillafilesystem_du': 'du',
    'gorillafilesystem_echo': 'echo',
    'gorillafilesystem_find': 'find',
    'gorillafilesystem_grep': 'grep',
    'gorillafilesystem_ls': 'ls',
    'gorillafilesystem_mkdir': 'mkdir',
    'gorillafilesystem_mv': 'mv',
    'gorillafilesystem_pwd': 'pwd',
    'gorillafilesystem_rm': 'rm',
    'gorillafilesystem_rmdir': 'rmdir',
    'gorillafilesystem_sort': 'sort',
    'gorillafilesystem_tail': 'tail',
    'gorillafilesystem_touch': 'touch',
    'gorillafilesystem_wc': 'wc',
    
    # TwitterAPI mappings (replaced PostingAPI)
    'TwitterAPI_post_tweet': 'post_tweet',
    'TwitterAPI_get_tweet': 'get_tweet',
    'TwitterAPI_delete_tweet': 'delete_tweet',
    'TwitterAPI_retweet': 'retweet',
    'TwitterAPI_comment': 'comment',
    'TwitterAPI_list_tweets': 'list_tweets',
    'TwitterAPI_authenticate_twitter': 'authenticate_twitter',
    'TwitterAPI_follow_user': 'follow_user',
    'TwitterAPI_unfollow_user': 'unfollow_user',
    'TwitterAPI_get_user_stats': 'get_user_stats',
    'TwitterAPI_search_tweets': 'search_tweets',
    
    # MathAPI mappings
    'MathAPI_logarithm': 'logarithm',
    'MathAPI_mean': 'mean',
    'MathAPI_standard_deviation': 'standard_deviation',
    'MathAPI_si_unit_conversion': 'si_unit_conversion',
    'MathAPI_imperial_si_conversion': 'imperial_si_conversion',
    'MathAPI_add': 'add',
    'MathAPI_subtract': 'subtract',
    'MathAPI_multiply': 'multiply',
    'MathAPI_divide': 'divide',
    'MathAPI_power': 'power',
    'MathAPI_square_root': 'square_root',
    'MathAPI_absolute_value': 'absolute_value',
    'MathAPI_round_number': 'round_number',
    'MathAPI_percentage': 'percentage',
    'MathAPI_min_value': 'min_value',
    'MathAPI_max_value': 'max_value',
    'MathAPI_sum_values': 'sum_values',
    
    # MessageAPI mappings
    'MessageAPI_list_users': 'list_users',
    'MessageAPI_get_user_id': 'get_user_id',
    'MessageAPI_message_login': 'message_login',
    'MessageAPI_message_get_login_status': 'message_get_login_status',
    'MessageAPI_send_message': 'send_message',
    'MessageAPI_delete_message': 'delete_message',
    'MessageAPI_view_messages_sent': 'view_messages_sent',
    'MessageAPI_add_contact': 'add_contact',
    'MessageAPI_search_messages': 'search_messages',
    'MessageAPI_get_message_stats': 'get_message_stats',
    
    # TicketAPI mappings
    'TicketAPI_create_ticket': 'create_ticket',
    'TicketAPI_get_ticket': 'get_ticket',
    'TicketAPI_close_ticket': 'close_ticket',
    'TicketAPI_resolve_ticket': 'resolve_ticket',
    'TicketAPI_edit_ticket': 'edit_ticket',
    'TicketAPI_ticket_login': 'ticket_login',
    'TicketAPI_ticket_get_login_status': 'ticket_get_login_status',
    'TicketAPI_logout': 'logout',
    'TicketAPI_get_user_tickets': 'get_user_tickets',
    
    # TradingBot mappings
    'TradingBot_get_current_time': 'get_current_time',
    'TradingBot_update_market_status': 'update_market_status',
    'TradingBot_get_symbol_by_name': 'get_symbol_by_name',
    'TradingBot_get_stock_info': 'get_stock_info',
    'TradingBot_get_order_details': 'get_order_details',
    'TradingBot_cancel_order': 'cancel_order',
    'TradingBot_place_order': 'place_order',
    'TradingBot_make_transaction': 'make_transaction',
    'TradingBot_get_account_info': 'get_account_info',
    'TradingBot_trading_login': 'trading_login',
    'TradingBot_trading_get_login_status': 'trading_get_login_status',
    'TradingBot_trading_logout': 'trading_logout',
    'TradingBot_fund_account': 'fund_account',
    'TradingBot_remove_stock_from_watchlist': 'remove_stock_from_watchlist',
    'TradingBot_get_watchlist': 'get_watchlist',
    'TradingBot_get_order_history': 'get_order_history',
    'TradingBot_get_transaction_history': 'get_transaction_history',
    'TradingBot_update_stock_price': 'update_stock_price',
    'TradingBot_get_available_stocks': 'get_available_stocks',
    'TradingBot_filter_stocks_by_price': 'filter_stocks_by_price',
    'TradingBot_add_to_watchlist': 'add_to_watchlist',
    'TradingBot_notify_price_change': 'notify_price_change',
    
    # TravelAPI mappings
    'TravelAPI_authenticate_travel': 'authenticate_travel',
    'TravelAPI_travel_get_login_status': 'travel_get_login_status',
    'TravelAPI_get_budget_fiscal_year': 'get_budget_fiscal_year',
    'TravelAPI_register_credit_card': 'register_credit_card',
    'TravelAPI_get_flight_cost': 'get_flight_cost',
    'TravelAPI_get_credit_card_balance': 'get_credit_card_balance',
    'TravelAPI_book_flight': 'book_flight',
    'TravelAPI_retrieve_invoice': 'retrieve_invoice',
    'TravelAPI_list_all_airports': 'list_all_airports',
    'TravelAPI_cancel_booking': 'cancel_booking',
    'TravelAPI_compute_exchange_rate': 'compute_exchange_rate',
    'TravelAPI_verify_traveler_information': 'verify_traveler_information',
    'TravelAPI_set_budget_limit': 'set_budget_limit',
    'TravelAPI_get_nearest_airport_by_city': 'get_nearest_airport_by_city',
    'TravelAPI_purchase_insurance': 'purchase_insurance',
    'TravelAPI_contact_customer_support': 'contact_customer_support',
    'TravelAPI_get_all_credit_cards': 'get_all_credit_cards',
    
    # VehicleControlAPI mappings
    'VehicleControlAPI_startEngine': 'startEngine',
    'VehicleControlAPI_fillFuelTank': 'fillFuelTank',
    'VehicleControlAPI_lockDoors': 'lockDoors',
    'VehicleControlAPI_adjustClimateControl': 'adjustClimateControl',
    'VehicleControlAPI_get_outside_temperature_from_google': 'get_outside_temperature_from_google',
    'VehicleControlAPI_get_outside_temperature_from_weather_com': 'get_outside_temperature_from_weather_com',
    'VehicleControlAPI_setHeadlights': 'setHeadlights',
    'VehicleControlAPI_displayCarStatus': 'displayCarStatus',
    'VehicleControlAPI_activateParkingBrake': 'activateParkingBrake',
    'VehicleControlAPI_pressBrakePedal': 'pressBrakePedal',
    'VehicleControlAPI_releaseBrakePedal': 'releaseBrakePedal',
    'VehicleControlAPI_setCruiseControl': 'setCruiseControl',
    'VehicleControlAPI_get_current_speed': 'get_current_speed',
    'VehicleControlAPI_display_log': 'display_log',
    'VehicleControlAPI_estimate_drive_feasibility_by_mileage': 'estimate_drive_feasibility_by_mileage',
    'VehicleControlAPI_liter_to_gallon': 'liter_to_gallon',
    'VehicleControlAPI_gallon_to_liter': 'gallon_to_liter',
    'VehicleControlAPI_estimate_distance': 'estimate_distance',
    'VehicleControlAPI_get_zipcode_based_on_city': 'get_zipcode_based_on_city',
    'VehicleControlAPI_set_navigation': 'set_navigation',
    
    # Also keep lowercase versions for backward compatibility
    'twitterapi_post_tweet': 'post_tweet',
    'twitterapi_get_tweet': 'get_tweet',
    'twitterapi_delete_tweet': 'delete_tweet',
    'twitterapi_retweet': 'retweet',
    'twitterapi_comment': 'comment',
    'twitterapi_list_tweets': 'list_tweets',
    'mathapi_logarithm': 'logarithm',
    'mathapi_mean': 'mean',
    'mathapi_standard_deviation': 'standard_deviation',
    'mathapi_si_unit_conversion': 'si_unit_conversion',
    'mathapi_imperial_si_conversion': 'imperial_si_conversion',
    'mathapi_add': 'add',
    'mathapi_subtract': 'subtract',
    'mathapi_multiply': 'multiply',
    'mathapi_divide': 'divide',
    'mathapi_power': 'power',
    'mathapi_square_root': 'square_root',
    'mathapi_absolute_value': 'absolute_value',
    'mathapi_round_number': 'round_number',
    'mathapi_percentage': 'percentage',
    'mathapi_min_value': 'min_value',
    'mathapi_max_value': 'max_value',
    'mathapi_sum_values': 'sum_values',
    'messageapi_list_users': 'list_users',
    'messageapi_get_user_id': 'get_user_id',
    'messageapi_message_login': 'message_login',
    'messageapi_message_get_login_status': 'message_get_login_status',
    'messageapi_send_message': 'send_message',
    'messageapi_delete_message': 'delete_message',
    'messageapi_view_messages_sent': 'view_messages_sent',
    'messageapi_add_contact': 'add_contact',
    'messageapi_search_messages': 'search_messages',
    'messageapi_get_message_stats': 'get_message_stats',
    'ticketapi_create_ticket': 'create_ticket',
    'ticketapi_get_ticket': 'get_ticket',
    'ticketapi_close_ticket': 'close_ticket',
    'ticketapi_resolve_ticket': 'resolve_ticket',
    'ticketapi_edit_ticket': 'edit_ticket',
    'ticketapi_ticket_login': 'ticket_login',
    'ticketapi_ticket_get_login_status': 'ticket_get_login_status',
    'ticketapi_logout': 'logout',
    'ticketapi_get_user_tickets': 'get_user_tickets',
    'tradingbot_get_current_time': 'get_current_time',
    'tradingbot_update_market_status': 'update_market_status',
    'tradingbot_get_symbol_by_name': 'get_symbol_by_name',
    'tradingbot_get_stock_info': 'get_stock_info',
    'tradingbot_get_order_details': 'get_order_details',
    'tradingbot_cancel_order': 'cancel_order',
    'tradingbot_place_order': 'place_order',
    'tradingbot_make_transaction': 'make_transaction',
    'tradingbot_get_account_info': 'get_account_info',
    'tradingbot_trading_login': 'trading_login',
    'tradingbot_trading_get_login_status': 'trading_get_login_status',
    'tradingbot_trading_logout': 'trading_logout',
    'tradingbot_fund_account': 'fund_account',
    'tradingbot_remove_stock_from_watchlist': 'remove_stock_from_watchlist',
    'tradingbot_get_watchlist': 'get_watchlist',
    'tradingbot_get_order_history': 'get_order_history',
    'tradingbot_get_transaction_history': 'get_transaction_history',
    'tradingbot_update_stock_price': 'update_stock_price',
    'tradingbot_get_available_stocks': 'get_available_stocks',
    'tradingbot_filter_stocks_by_price': 'filter_stocks_by_price',
    'tradingbot_add_to_watchlist': 'add_to_watchlist',
    'tradingbot_notify_price_change': 'notify_price_change',
    'travelapi_authenticate_travel': 'authenticate_travel',
    'travelapi_travel_get_login_status': 'travel_get_login_status',
    'travelapi_get_budget_fiscal_year': 'get_budget_fiscal_year',
    'travelapi_register_credit_card': 'register_credit_card',
    'travelapi_get_flight_cost': 'get_flight_cost',
    'travelapi_get_credit_card_balance': 'get_credit_card_balance',
    'travelapi_book_flight': 'book_flight',
    'travelapi_retrieve_invoice': 'retrieve_invoice',
    'travelapi_list_all_airports': 'list_all_airports',
    'travelapi_cancel_booking': 'cancel_booking',
    'travelapi_compute_exchange_rate': 'compute_exchange_rate',
    'travelapi_verify_traveler_information': 'verify_traveler_information',
    'travelapi_set_budget_limit': 'set_budget_limit',
    'travelapi_get_nearest_airport_by_city': 'get_nearest_airport_by_city',
    'travelapi_purchase_insurance': 'purchase_insurance',
    'travelapi_contact_customer_support': 'contact_customer_support',
    'travelapi_get_all_credit_cards': 'get_all_credit_cards',
    'vehiclecontrolapi_startengine': 'startEngine',
    'vehiclecontrolapi_fillfueltank': 'fillFuelTank',
    'vehiclecontrolapi_lockdoors': 'lockDoors',
    'vehiclecontrolapi_adjustclimatecontrol': 'adjustClimateControl',
    'vehiclecontrolapi_get_outside_temperature_from_google': 'get_outside_temperature_from_google',
    'vehiclecontrolapi_get_outside_temperature_from_weather_com': 'get_outside_temperature_from_weather_com',
    'vehiclecontrolapi_setheadlights': 'setHeadlights',
    'vehiclecontrolapi_displaycarstatus': 'displayCarStatus',
    'vehiclecontrolapi_activateparkingbrake': 'activateParkingBrake',
    'vehiclecontrolapi_pressbrakepedal': 'pressBrakePedal',
    'vehiclecontrolapi_releasebrakepedal': 'releaseBrakePedal',
    'vehiclecontrolapi_setcruisecontrol': 'setCruiseControl',
    'vehiclecontrolapi_get_current_speed': 'get_current_speed',
    'vehiclecontrolapi_display_log': 'display_log',
    'vehiclecontrolapi_estimate_drive_feasibility_by_mileage': 'estimate_drive_feasibility_by_mileage',
    'vehiclecontrolapi_liter_to_gallon': 'liter_to_gallon',
    'vehiclecontrolapi_gallon_to_liter': 'gallon_to_liter',
    'vehiclecontrolapi_estimate_distance': 'estimate_distance',
    'vehiclecontrolapi_get_zipcode_based_on_city': 'get_zipcode_based_on_city',
    'vehiclecontrolapi_set_navigation': 'set_navigation',
}


class BFCLBenchmark(BaseBenchmark):
    
    def __init__(self, 
                 data_dir: Optional[str] = None,
                 ground_truth_dir: Optional[str] = None,
                 **config):
        super().__init__(**config)
        
        self.loader = BFCLTestLoader(data_dir, ground_truth_dir)
        
        logger.info("BFCL Benchmark plugin initialized")
        
        self._function_name_cache = {}
        self._normalized_mapping_cache = {}
    
    @property
    def name(self) -> str:
        return "bfcl"
    
    @property
    def version(self) -> str:
        return "3.0"
    
    def list_test_ids(self, **filters) -> List[str]:
        category = filters.get('category')
        limit = filters.get('limit')
        
        test_ids = self.loader.list_test_ids(category)
        
        if limit and isinstance(limit, int):
            test_ids = test_ids[:limit]
        
        return test_ids
    
    def load_test_case(self, test_id: str) -> TestCase:
        bfcl_test_case = self.loader.load_test_case(test_id)
        
        metadata = {
            'category': self._extract_category(test_id),
            'involved_classes': getattr(bfcl_test_case, 'involved_classes', []),
            'initial_config': getattr(bfcl_test_case, 'initial_config', {}),
            'bfcl_original': True
        }
        
        # Priority:
        # 1) category/id naming (authoritative for BFCL multi_turn_* categories)
        # 2) question shape fallback
        question = getattr(bfcl_test_case, 'question', [])
        category = metadata.get('category') or ''
        is_multi_turn = False
        if isinstance(category, str) and category.startswith('multi_turn'):
            is_multi_turn = True
        elif isinstance(test_id, str) and test_id.startswith('multi_turn'):
            is_multi_turn = True
        elif isinstance(question, list) and len(question) > 1:
            is_multi_turn = True

        if is_multi_turn:
            metadata['type'] = 'multi_turn'
            metadata['turns'] = len(question) if isinstance(question, list) and len(question) > 0 else 1
        else:
            metadata['type'] = 'single_turn'
        
        return TestCase(
            id=test_id,
            metadata=metadata,
            content=bfcl_test_case,
            expected_outputs=None,
            evaluation_config={}
        )
    
    def create_evaluator(self) -> BFCLEvaluator:
        return BFCLEvaluator(self.loader)
    
    def format_result(self, 
                     test_case: TestCase, 
                     execution_result: ExecutionResult, 
                     evaluation: Dict[str, Any]) -> Dict[str, Any]:
        function_info = self._extract_function_definitions(test_case)
        
        turns_data = self._extract_turns_data(test_case, execution_result)

        result = {
            'id': TestCaseAdapter.get_id(test_case),
            'benchmark': self.name,
            'category': test_case.get_category(),
            'function_definitions': function_info,
            'execution_summary': {
                'success': execution_result.success,
                'total_execution_time': execution_result.get_execution_time(),
                'total_attempts': self._count_total_attempts(execution_result.metadata, turns_data),
                'final_score': execution_result.metadata.get('final_score', 0.0),
                'real_execution_enabled': execution_result.metadata.get('real_execution_enabled', False),
                'total_turns': execution_result.metadata.get('total_turns', test_case.metadata.get('turns', 1)),
                'executed_turns': execution_result.metadata.get('executed_turns', test_case.metadata.get('turns', 1)),
                'max_turns_applied': execution_result.metadata.get('max_turns_applied')
            }
        }
        result['turns'] = turns_data
        result['execution'] = result['execution_summary']

        process_trace = execution_result.metadata.get('process_trace')
        if isinstance(process_trace, dict) and process_trace:
            result['process_trace'] = process_trace

        return result
    
    def _extract_function_definitions(self, test_case: TestCase) -> List[Dict[str, Any]]:
        function_info = None
        
        if hasattr(test_case, 'content'):
            content = test_case.content
            if hasattr(content, 'function'):
                function_info = content.function
            elif hasattr(content, 'functions'):
                function_info = content.functions
        
        if function_info is None:
            return []
        elif isinstance(function_info, list):
            return function_info
        elif isinstance(function_info, dict):
            return list(function_info.values()) if function_info else []
        else:
            try:
                if hasattr(function_info, '__iter__') and not isinstance(function_info, str):
                    return list(function_info)
                else:
                    return [function_info]
            except (TypeError, ValueError):
                logger.warning(f"Unable to convert function_info to list: {type(function_info)}")
                return []
    
    def _count_total_attempts(
        self,
        metadata: Dict[str, Any],
        turns_data: Optional[List[Dict[str, Any]]] = None,
    ) -> int:
        if turns_data:
            total = 0
            for turn in turns_data:
                retries = turn.get('retries', []) if isinstance(turn, dict) else []
                if isinstance(retries, list) and retries:
                    total += len(retries)
            if total > 0:
                return total

        attempt_keys = [key for key in metadata.keys() if key.startswith('attempt_') and key.endswith('_evaluation')]
        return len(attempt_keys) if attempt_keys else 1
    
    def _extract_turns_data(self, test_case: TestCase, execution_result: ExecutionResult) -> List[Dict[str, Any]]:
        turns = []
        
        is_multi_turn = test_case.metadata.get('type') == 'multi_turn'
        
        if is_multi_turn:
            turn_results = []
            for output in execution_result.outputs:
                if isinstance(output, dict) and output.get('type') == 'turn_result':
                    turn_results.append(output)
            
            turn_results.sort(key=lambda x: x.get('turn_index', 0))
            
            for turn_idx, turn_output in enumerate(turn_results):
                metadata = turn_output.get('content', {}).copy()
                
                # Debug logging to check if agent_response is in metadata
                for key in metadata.keys():
                    if 'agent_response' in key or 'tools_count' in key:
                        logger.debug(f"[DEBUG EXTRACT_TURNS] Found key in metadata: {key}")
                
                if 'mock_tool_calls' in turn_output:
                    metadata['mock_tool_calls'] = turn_output['mock_tool_calls']
                if 'real_tool_calls' in turn_output:
                    metadata['real_tool_calls'] = turn_output['real_tool_calls']
                if 'mock_config' in turn_output:
                    metadata['mock_config'] = turn_output['mock_config']
                if 'calibrated_config' in turn_output:
                    metadata['calibrated_config'] = turn_output['calibrated_config']
                
                turn_data = self._process_single_turn_data(
                    turn_idx, 
                    test_case, 
                    metadata,
                    is_multi_turn=True,
                    turn_output=turn_output
                )
                turns.append(turn_data)
        else:
            turn_data = self._process_single_turn_data(
                0, 
                test_case, 
                execution_result.metadata,
                is_multi_turn=False
            )
            # Keep the canonical final answer on the formatted turn object so
            # main-result export and BFCL-eval export can share one source.
            if execution_result.tool_calls:
                turn_data['final_tool_calls'] = execution_result.tool_calls
            turns.append(turn_data)
        
        return turns
    
    def _process_single_turn_data(self, turn_idx: int, test_case: TestCase, 
                                 metadata: Dict[str, Any], is_multi_turn: bool = False,
                                 turn_output: Dict[str, Any] = None) -> Dict[str, Any]:
        question = self._extract_question_for_turn(test_case, turn_idx, is_multi_turn)
        
        attempts_data = self._extract_attempts_data(metadata)
        
        checklist = self._extract_checklist_for_turn(metadata)
        
        turn_data = {
            'turn_id': turn_idx,
            'question': question,
            'checklist': checklist,
            'retries': attempts_data
        }
        
        if 'calibrated_config' in metadata:
            turn_data['calibrated_config'] = metadata['calibrated_config']
        
        if 'real_tool_calls' in metadata:
            turn_data['real_tool_calls'] = metadata['real_tool_calls']
        
        return turn_data
    
    def _extract_question_for_turn(self, test_case: TestCase, turn_idx: int, is_multi_turn: bool) -> str:
        if not hasattr(test_case, 'content') or not hasattr(test_case.content, 'question'):
            return "No question available"
        
        raw_question = test_case.content.question
        
        if is_multi_turn and isinstance(raw_question, list) and len(raw_question) > turn_idx:
            turn_question = raw_question[turn_idx]
            return self._extract_question_content(turn_question)
        elif not is_multi_turn:
            if isinstance(raw_question, list) and len(raw_question) > 0:
                return self._extract_question_content(raw_question[0])
            else:
                return str(raw_question)
        
        return "No question available"
    
    def _extract_question_content(self, question_data) -> str:
        if isinstance(question_data, str):
            return question_data
        elif isinstance(question_data, list) and len(question_data) > 0:
            first_item = question_data[0]
            if isinstance(first_item, dict) and 'content' in first_item:
                return first_item['content']
            else:
                return str(first_item)
        elif isinstance(question_data, dict) and 'content' in question_data:
            return question_data['content']
        else:
            return str(question_data)
    
    def _extract_checklist_for_turn(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        attempt_keys = [key for key in metadata.keys() if key.startswith('attempt_') and key.endswith('_evaluation')]
        if not attempt_keys:
            return []
        
        attempt_keys.sort(key=lambda x: int(x.split('_')[1]))
        first_attempt_key = attempt_keys[0]
        first_evaluation = metadata.get(first_attempt_key, {})
        
        checklist = first_evaluation.get('checklist', [])
        
        if checklist is None:
            checklist = []
        elif not isinstance(checklist, list):
            checklist = []
        
        formatted_checklist = []
        for idx, item in enumerate(checklist):
            if isinstance(item, dict):
                formatted_item = item.copy()
                formatted_item['id'] = idx
                formatted_checklist.append(formatted_item)
            else:
                formatted_checklist.append({
                    'id': idx,
                    'description': str(item)
                })
        
        return formatted_checklist
    
    def _extract_attempts_data(self, metadata: Dict[str, Any]) -> List[Dict[str, Any]]:
        attempt_keys = [key for key in metadata.keys() if key.startswith('attempt_') and key.endswith('_evaluation')]
        if not attempt_keys:
            return []
        
        attempt_keys.sort(key=lambda x: int(x.split('_')[1]))
        attempts = []
        
        for attempt_key in attempt_keys:
            attempt_num = int(attempt_key.split('_')[1])
            evaluation_data = metadata.get(attempt_key, {})
            tool_calls_key = f'attempt_{attempt_num}_tool_calls'
            tool_calls_data = metadata.get(tool_calls_key, [])
            
            attempt_data = {
                'attempt': attempt_num,
                'tool_calls': tool_calls_data,
                'judgment': {
                    'score': evaluation_data.get('score', 0),
                    'passed': evaluation_data.get('passed', False)
                }
            }
            
            # Add mock_config if available
            mock_config_key = f'attempt_{attempt_num}_mock_config'
            if mock_config_key in metadata:
                attempt_data['mock_config'] = metadata.get(mock_config_key, {})
                logger.debug(f"[DEBUG BENCHMARK] Found mock_config for attempt {attempt_num}")
            
            # Add agent_response if available
            agent_response_key = f'attempt_{attempt_num}_agent_response'
            if agent_response_key in metadata:
                attempt_data['agent_response'] = metadata.get(agent_response_key, '')
                logger.debug(f"[DEBUG BENCHMARK] Found agent_response for attempt {attempt_num}: {len(metadata.get(agent_response_key, ''))} chars")
            
            # Add tools_count if available
            tools_count_key = f'attempt_{attempt_num}_tools_count'
            if tools_count_key in metadata:
                attempt_data['tools_count'] = metadata.get(tools_count_key, 0)
                logger.debug(f"[DEBUG BENCHMARK] Found tools_count for attempt {attempt_num}: {metadata.get(tools_count_key, 0)}")
            
            # Note: real_tool_calls, calibrated_config, and mock_config are recorded at turn level
            # to avoid duplication. They apply to the last successful attempt only.
            
            # Add feedback if available
            if 'feedback' in evaluation_data:
                attempt_data['judgment']['feedback'] = evaluation_data['feedback']
            
            # Add retry_reason if available 
            if 'retry_reason' in evaluation_data and evaluation_data['retry_reason']:
                attempt_data['retry_context'] = evaluation_data['retry_reason']
            elif attempt_num > 0:
                # For backwards compatibility, add a generic retry context
                attempt_data['retry_context'] = f"Retry attempt {attempt_num} based on previous failure"
            
            if not attempt_data['judgment']['passed'] and evaluation_data.get('retry_reason'):
                attempt_data['judgment']['retry_reason'] = evaluation_data['retry_reason']
            
            if attempt_num > 0:
                attempt_data['retry_context'] = f"Retry attempt {attempt_num} based on previous failure"
            
            judgment_data = evaluation_data.get('judgment', [])
            if judgment_data:
                checklist_results = []
                for idx, judgment_item in enumerate(judgment_data):
                    if isinstance(judgment_item, dict):
                        checklist_results.append({
                            'checklist_id': idx,
                            'status': judgment_item.get('status', 'unknown'),
                            'reasoning': judgment_item.get('reasoning', '')
                        })
                
                attempt_data['judgment']['checklist_results'] = checklist_results
            
            attempts.append(attempt_data)
        
        return attempts
    
    def _summarize_judgment(self, judgment_list: List[Dict]) -> Dict[str, Any]:
        if not judgment_list:
            return {}
        
        success_count = sum(1 for j in judgment_list if j.get('status') == 'success')
        total_count = len(judgment_list)
        
        return {
            'total': total_count,
            'success': success_count,
            'success_rate': success_count / total_count if total_count > 0 else 0
        }
    
    def _extract_category(self, test_id: str) -> str:
        if '_' in test_id:
            parts = test_id.split('_')
            if len(parts) >= 2:
                # multi_turn_base_0 -> multi_turn_base
                if parts[0] == 'multi' and len(parts) >= 3:
                    return f"{parts[0]}_{parts[1]}_{parts[2]}"
                # live_simple_5 -> live_simple
                elif parts[0] == 'live' and len(parts) >= 3:
                    return f"{parts[0]}_{parts[1]}"
                # simple_0 -> simple
                else:
                    return parts[0]
        
        return 'unknown'
    
    def list_categories(self) -> List[str]:
        return self.loader.list_categories()
    
    def get_category_stats(self, category: str) -> Dict[str, Any]:
        test_ids = self.list_test_ids(category=category)
        
        type_counts = {}
        for test_id in test_ids:
            try:
                test_case = self.load_test_case(test_id)
                test_type = test_case.metadata.get('type', 'unknown')
                type_counts[test_type] = type_counts.get(test_type, 0) + 1
            except Exception:
                type_counts['error'] = type_counts.get('error', 0) + 1
        
        return {
            'category': category,
            'total_tests': len(test_ids),
            'type_distribution': type_counts,
            'first_test': test_ids[0] if test_ids else None,
            'last_test': test_ids[-1] if test_ids else None
        }
    
    def validate_test_id(self, test_id: str) -> bool:
        return self.loader.test_case_exists(test_id)
    
    def get_test_case_info(self, test_id: str) -> Dict[str, Any]:
        if not self.validate_test_id(test_id):
            return {'exists': False}
        
        try:
            test_case = self.load_test_case(test_id)
            return {
                'exists': True,
                'id': TestCaseAdapter.get_id(test_case),
                'category': test_case.get_category(),
                'type': test_case.metadata.get('type'),
                'turns': test_case.metadata.get('turns', 1),
                'involved_classes': test_case.metadata.get('involved_classes', []),
                'has_initial_config': bool(test_case.metadata.get('initial_config'))
            }
        except Exception as e:
            return {'exists': True, 'error': str(e)}
    
    def get_benchmark_info(self) -> Dict[str, Any]:
        base_info = super().get_benchmark_info()
        
        categories = self.list_categories()
        
        bfcl_info = {
            'description': 'Berkeley Function Call Leaderboard - Evaluating LLMs\' ability to call functions',
            'categories': categories,
            'total_categories': len(categories),
            'supports_multi_turn': True,
            'supports_tool_execution': True,
            'data_source': 'Berkeley Function Call Leaderboard'
        }
        
        base_info.update(bfcl_info)
        
        return base_info
    
    def get_original_function_names(self, test_id: str) -> List[str]:
        if test_id in self._function_name_cache:
            return self._function_name_cache[test_id]
        
        try:
            test_case = self.load_test_case(test_id)
            
            function_names = []
            if hasattr(test_case.content, 'function') and test_case.content.function:
                for func in test_case.content.function:
                    if isinstance(func, dict) and 'name' in func:
                        function_names.append(func['name'])
            
            self._function_name_cache[test_id] = function_names
            logger.debug(f"Cached function names for {test_id}: {function_names}")
            
            return function_names
            
        except Exception as e:
            logger.warning(f"Failed to get function names for {test_id}: {e}")
            return []
    
    def _is_multi_turn_test(self, test_id: str) -> bool:
        return test_id.startswith('multi_turn_')
    
    def _normalize_function_name(self, name: str) -> str:
        import re
        if name.startswith('/'):
            name = name[1:]
        normalized = re.sub(r'[^a-zA-Z0-9]+', '_', name)
        normalized = normalized.strip('_')
        return normalized.lower()

    def _strip_runtime_function_prefix(self, tool_call_function: str) -> str:
        """
        Remove runtime-generated prefixes while preserving the operation body.

        Examples:
            live_multiple_126-48-0_1post_analysis_api_AnalysisApi_retrieve_
              -> analysis_api_AnalysisApi_retrieve_
            /live_multiple_244-108-0_0post_version_api_VersionApi_get_versio
              -> version_api_VersionApi_get_versio
            LM991_WebsiteConfigurationApi_rename_website
              -> WebsiteConfigurationApi_rename_website
        """
        import re

        cleaned = tool_call_function[1:] if tool_call_function.startswith('/') else tool_call_function
        cleaned = re.sub(r'^[A-Za-z_]+(?:_\d+(?:-\d+)*?)?_\d+post_', '', cleaned)
        cleaned = re.sub(r'^[A-Z]{1,4}\d+_', '', cleaned)
        return cleaned

    def _recover_truncated_function_name(self, test_id: str, tool_call_function: str) -> Optional[str]:
        """
        Recover a likely original function name when the runtime function name was truncated.

        The recovery is constrained to functions defined in the current test case.
        It only succeeds when there is a single unambiguous candidate whose normalized
        original name starts with the normalized stripped runtime name.
        """
        stripped = self._strip_runtime_function_prefix(tool_call_function)
        normalized_raw = self._normalize_function_name(stripped)
        if not normalized_raw:
            return None

        candidates = []
        for original_name in self.get_original_function_names(test_id):
            normalized_original = self._normalize_function_name(original_name)
            if not normalized_original:
                continue
            if (
                normalized_original.startswith(normalized_raw)
                or normalized_original.endswith(normalized_raw)
            ):
                candidates.append(original_name)

        if len(candidates) == 1:
            logger.debug(
                f"Recovered truncated function name: {tool_call_function} -> {candidates[0]}"
            )
            return candidates[0]

        if len(candidates) > 1:
            logger.warning(
                "Ambiguous truncated function name for %s in %s: %s",
                tool_call_function,
                test_id,
                candidates,
            )
        return None
    
    def _build_normalized_mapping(self, test_id: str) -> Dict[str, str]:
        if test_id in self._normalized_mapping_cache:
            return self._normalized_mapping_cache[test_id]
        
        mapping = {}
        original_functions = self.get_original_function_names(test_id)
        
        for original_name in original_functions:
            normalized = self._normalize_function_name(original_name)
            mapping[normalized] = original_name
            logger.debug(f"Mapping {normalized} -> {original_name}")
        
        self._normalized_mapping_cache[test_id] = mapping
        return mapping

    def _build_runtime_mapping(self, test_id: str) -> Dict[str, str]:
        mapping: Dict[str, str] = {}
        for original_name in self.get_original_function_names(test_id):
            runtime_name = derive_single_turn_runtime_function_name(test_id, original_name)
            mapping[runtime_name] = original_name
            mapping[self._normalize_function_name(runtime_name)] = original_name
        return mapping
    
    def map_tool_call_to_original_function(self, test_id: str, tool_call_function: str) -> str:
        # Check if this is a multi-turn test
        if self._is_multi_turn_test(test_id):
            # Clean the function name (remove leading slash if present)
            clean_name = tool_call_function
            if clean_name.startswith('/'):
                clean_name = clean_name[1:]
            
            # Try direct mapping from the multi-turn table (exact match first)
            if clean_name in MULTI_TURN_FUNCTION_MAPPING:
                logger.debug(f"Multi-turn mapping (exact): {tool_call_function} -> {MULTI_TURN_FUNCTION_MAPPING[clean_name]}")
                return MULTI_TURN_FUNCTION_MAPPING[clean_name]
            
            # Try lowercase version for backward compatibility
            clean_name_lower = clean_name.lower()
            if clean_name_lower in MULTI_TURN_FUNCTION_MAPPING:
                logger.debug(f"Multi-turn mapping (lowercase): {tool_call_function} -> {MULTI_TURN_FUNCTION_MAPPING[clean_name_lower]}")
                return MULTI_TURN_FUNCTION_MAPPING[clean_name_lower]
            
            # If not found directly, try removing common prefixes (both capital and lowercase)
            api_prefixes = [
                # Lowercase versions (for backward compatibility)
                'gorillafilesystem_', 'twitterapi_', 'mathapi_', 
                'messageapi_', 'ticketapi_', 'tradingbot_', 
                'travelapi_', 'vehiclecontrolapi_',
                # Capital versions (new naming convention)
                'GorillaFileSystem_', 'TwitterAPI_', 'MathAPI_', 
                'MessageAPI_', 'TicketAPI_', 'TradingBot_', 
                'TravelAPI_', 'VehicleControlAPI_'
            ]
            
            for prefix in api_prefixes:
                # Check both original case and lowercase
                if clean_name.startswith(prefix):
                    # First check if the full name with prefix is in the mapping
                    if clean_name in MULTI_TURN_FUNCTION_MAPPING:
                        return MULTI_TURN_FUNCTION_MAPPING[clean_name]
                    # Otherwise return the base name without prefix
                    base_name = clean_name[len(prefix):]
                    logger.debug(f"Multi-turn prefix removal: {tool_call_function} -> {base_name}")
                    return base_name
                elif clean_name_lower.startswith(prefix.lower()):
                    # Check lowercase version
                    if clean_name_lower in MULTI_TURN_FUNCTION_MAPPING:
                        return MULTI_TURN_FUNCTION_MAPPING[clean_name_lower]
                    # Otherwise return the base name without prefix
                    base_name = clean_name_lower[len(prefix.lower()):]
                    logger.debug(f"Multi-turn prefix removal (lowercase): {tool_call_function} -> {base_name}")
                    return base_name
            
            # If no known prefix, return the cleaned name (without leading slash)
            logger.debug(f"Multi-turn no mapping found, using cleaned: {tool_call_function} -> {clean_name}")
            return clean_name
        
        # Original single-turn logic
        runtime_mapping = self._build_runtime_mapping(test_id)
        if tool_call_function in runtime_mapping:
            return runtime_mapping[tool_call_function]

        normalized_tool_call = self._normalize_function_name(tool_call_function)
        if normalized_tool_call in runtime_mapping:
            return runtime_mapping[normalized_tool_call]

        mapping = self._build_normalized_mapping(test_id)
        
        if normalized_tool_call in mapping:
            return mapping[normalized_tool_call]
        
        for normalized_name, original_name in mapping.items():
            if normalized_tool_call in normalized_name or normalized_name in normalized_tool_call:
                logger.debug(f"Partial match: {tool_call_function} -> {original_name}")
                mapping[normalized_tool_call] = original_name
                return original_name

        recovered_name = self._recover_truncated_function_name(test_id, tool_call_function)
        if recovered_name:
            mapping[normalized_tool_call] = recovered_name
            return recovered_name
        
        logger.warning(f"Could not map {tool_call_function} to any original function in {test_id}")
        cleaned = tool_call_function
        if cleaned.startswith('/'):
            cleaned = cleaned[1:]
        import re
        cleaned = re.sub(r'^[a-z_]+_\d+_\d+/', '', cleaned)
        return cleaned
    
    def _clean_function_name(self, tool_call_function: str) -> str:
        if tool_call_function.startswith('/'):
            tool_call_function = tool_call_function[1:]
        
        patterns_to_remove = [
            r'^[a-z_]+_\d+_\d+/',
            r'simple_\d+_\d+post_',
            r'multiple_\d+_\d+post_',
            r'parallel_\d+_\d+post_',
            r'live_simple_\d+-\d+-\d+_\d+post_',
            r'live_multiple_\d+-\d+-\d+_\d+post_',
            r'multi_turn_\w+_\d+_\d+post_',
            r'_\d+_\d+post_',
            r'post_'
        ]
        
        import re
        for pattern in patterns_to_remove:
            tool_call_function = re.sub(pattern, '', tool_call_function)
        
        original_with_dots = tool_call_function
        
        tool_call_function = tool_call_function.replace('/', '_')
        
        if '_' in tool_call_function and '.' not in tool_call_function:
            parts = tool_call_function.split('_', 1)
            if len(parts) == 2:
                return f"{parts[0]}.{parts[1]}"
        
        return tool_call_function
