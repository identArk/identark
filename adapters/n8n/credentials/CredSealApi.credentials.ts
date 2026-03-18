import {
	IAuthenticateGeneric,
	ICredentialTestRequest,
	ICredentialType,
	INodeProperties,
} from 'n8n-workflow';

export class CredSealApi implements ICredentialType {
	name = 'credSealApi';
	displayName = 'CredSeal API';
	documentationUrl = 'https://github.com/credseal/sdk';
	properties: INodeProperties[] = [
		{
			displayName: 'API Key',
			name: 'apiKey',
			type: 'string',
			typeOptions: { password: true },
			default: '',
			required: true,
			description: 'Your CredSeal API key (starts with csk_)',
		},
		{
			displayName: 'Control Plane URL',
			name: 'baseUrl',
			type: 'string',
			default: 'https://credseal-cloud.fly.dev',
			required: true,
			description: 'CredSeal control plane URL',
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
