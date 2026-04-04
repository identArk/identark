import {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class IdentArkApi implements ICredentialType {
	name = 'identarkApi';
	displayName = 'IdentArk API';
	documentationUrl = 'https://github.com/identark/sdk';
	properties: INodeProperties[] = [
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			required: true,
			description: 'Your IdentArk API key (starts with iak_)',
		},
		{
			displayName: 'Control Plane URL',
			name: 'baseUrl',
			type: 'string',
			default: 'https://identark-cloud.fly.dev',
			required: true,
			description: 'IdentArk control plane URL',
		},
	];

	authenticate: IAuthenticateGeneric = {
		type: 'generic',
		properties: {
			headers: {
				Authorization: '={{"Bearer " + $credentials.apiKey}}',
			},
		},
	};

	test: ICredentialTestRequest = {
		request: {
			baseURL: '={{$credentials.baseUrl}}',
			url: '/health',
			method: 'GET',
		},
	};
}
