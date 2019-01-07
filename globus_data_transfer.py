import logging
from globus_sdk import NativeAppAuthClient, TransferClient, TransferData, RefreshTokenAuthorizer

CLIENT_ID = '4674f980-e5c3-4172-8fdb-7f40a9fb3055'


class Transfer:
    '''
    Modified Transfer, add an option to pass refresh token to avoid the web login
    '''
    def __init__(self, src_endpoint_name, dst_endpoint_name, transfer_rt=None, log_lv=logging.INFO):
        log_format = '%(asctime)-15s %(levelname)s:\t  class:%(name)s %(message)s'
        logging.basicConfig(format=log_format)
        self.logger = logging.getLogger(self.__class__.__name__)
        self.logger.setLevel(log_lv)
        self.logger.debug('CLIENT_ID: {0}'.format(CLIENT_ID))
        self.client = NativeAppAuthClient(CLIENT_ID)
        self.client.oauth2_start_flow(refresh_tokens=True)
        
        if transfer_rt is not None:
            self.authorizer = RefreshTokenAuthorizer(transfer_rt, self.client)
        else:
            authorize_url = self.client.oauth2_get_authorize_url()
            print('Please go to this URL and login: {0}'.format(authorize_url))

            get_input = getattr(__builtins__, 'raw_input', input)
            auth_code = get_input(
                'Please enter the code you get after login here: ').strip()
            token_response = self.client.oauth2_exchange_code_for_tokens(auth_code)
            self.globus_auth_data = token_response.by_resource_server['auth.globus.org']
            self.globus_transfer_data = token_response.by_resource_server['transfer.api.globus.org']
            auth_token = self.globus_auth_data['access_token']
            transfer_token = self.globus_transfer_data['access_token']
            transfer_rt = self.globus_transfer_data['refresh_token']
            transfer_at = self.globus_transfer_data['access_token']
            expires_at_s = self.globus_transfer_data['expires_at_seconds']
            self.authorizer = RefreshTokenAuthorizer(transfer_rt, self.client,
                                                     access_token=transfer_at,
                                                     expires_at=expires_at_s)
        self.transferClient = TransferClient(authorizer=self.authorizer)
        self.src_endpoint = None
        self.dst_endpoint = None
        for ep in self.transferClient.endpoint_search(filter_scope="shared-with-me"):
            if ep["display_name"] == src_endpoint_name:
                self.src_endpoint = ep
                self.logger.info('Source endpoint: [{0}] {1}'
                                 .format(self.src_endpoint['id'], self.src_endpoint['display_name']))
        if self.src_endpoint is None:
            self.logger.error('No endpoint shared with you with name: {0}'.format(src_endpoint_name))
            raise LookupError
        for ep in self.transferClient.endpoint_search(filter_scope="my-endpoints"):
            if ep['display_name'] == dst_endpoint_name:
                self.dst_endpoint = ep
                self.logger.info('Destination endpoint: [{0}] {1}'
                                 .format(self.dst_endpoint['id'], self.dst_endpoint['display_name']))
        if self.dst_endpoint is None:
            self.logger.error('You don\'t have endpoint named: {0}'.format(dst_endpoint_name))
            raise LookupError

    def transfer_dir(self, src_dir, dst_dir):
        transfer_data = TransferData(self.transferClient, self.src_endpoint['id'], self.dst_endpoint['id'])
        transfer_data.add_item(src_dir, dst_dir, recursive=True)
        result = self.transferClient.submit_transfer(transfer_data)
        self.logger.info('task [{0}] {1}'.format(result['task_id'], result['code']))
        return result

    def transfer_file(self, src_file, dst_file):
        transfer_data = TransferData(self.transferClient, self.src_endpoint['id'], self.dst_endpoint['id'])
        transfer_data.add_item(src_file, dst_file)
        result = self.transferClient.submit_transfer(transfer_data)
        self.logger.info('task_id [{0}] {1}'.format(result['task_id'], result['code']))
        return result

    def ls_src_dir(self, path, ls_filter=''):
        # using iteration to get every entry from result
        # an entry contain two keys: 'name' and 'type'
        # type define the entry is a file or folder
        result = self.transferClient.operation_ls(self.src_endpoint['id'], path=path, filter=ls_filter)
        for entry in result:
            self.logger.debug('name: {0}\ttype: {1}'.format(entry["name"], entry["type"]))
        return result

    def task_list(self, num_results=10):
        result = self.transferClient.task_list(num_results=num_results)
        for task in result:
            self.logger.debug('task_id: [{0}]\t status: {1}'.format(task['task_id'], task['status']))
            
        result = self.transferClient.task_list(num_results=num_results)
        return result
    def get_task(self, task_id):
        return self.transferClient.get_task(task_id)

if __name__ == '__main__':
    # demo
    # init session
    t = Transfer('Terraref', 'Lab Server', log_lv=logging.DEBUG)
    # list files under the dir
    file_list = t.ls_src_dir('/ua-mac/raw_data/scanner3DTop/2017-04-30/')
    # transfer a dir to the dst endpoint
    # change dst dir
    t.transfer_dir('/ua-mac/Level_1/scanner3DTop/2017-04-30/',
                   '/media/zli/data/Terra/globus_transfer_test/2017-04-30/')
    # get task status
    task_list = t.task_list()